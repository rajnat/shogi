use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use clap::{Parser, Subcommand};
use shogi_core::board::Board;
use shogi_core::perft::{perft, perft_divide};
use shogi_core::usi::run_usi_loop;

#[derive(Parser)]
#[command(name = "shogi", version, about = "Shogi engine (USI compatible)")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the USI protocol loop (default when invoked without arguments)
    Usi,
    /// Run perft from startpos
    Perft {
        #[arg(short, long, default_value_t = 5)]
        depth: u32,
        /// Print per-move counts
        #[arg(short = 'D', long)]
        divide: bool,
    },
    /// Print the startpos SFEN string
    Startpos,
    /// Run the AlphaZero self-play training loop
    Train {
        /// Network channel width (8 = quick smoke-test; 256 = full AlphaZero)
        #[arg(long, default_value_t = 256)]
        channels: i64,
        /// Number of residual blocks (2 = quick; 20 = full AlphaZero)
        #[arg(long, default_value_t = 20)]
        blocks: usize,
        /// Number of parallel self-play worker threads
        #[arg(long, default_value_t = 4)]
        workers: usize,
        /// Directory to write checkpoint files into
        #[arg(long, default_value = "checkpoints")]
        checkpoint_dir: String,
        /// Save a checkpoint every N training steps (0 = never)
        #[arg(long, default_value_t = 1000)]
        checkpoint_every: u64,
        /// Stop after N total training steps (0 = run until Ctrl-C)
        #[arg(long, default_value_t = 0)]
        total_steps: u64,
        /// Number of pit games to play after each checkpoint (0 = skip)
        #[arg(long, default_value_t = 100)]
        pit_games: u64,
        /// Resume training from this checkpoint file
        #[arg(long)]
        resume: Option<PathBuf>,
    },
    /// Benchmark neural-net MCTS vs rollout-MCTS baseline
    ///
    /// Plays a match between two MCTS agents: one using network policy+value,
    /// one using uniform prior + random rollouts.  Colors alternate every game.
    ///
    /// With no --checkpoint the network has random (untrained) weights, so
    /// both agents will play at roughly equal strength.  Load a trained
    /// checkpoint to see the real improvement.
    Bench {
        /// Number of games in the match (must be even for balanced colors)
        #[arg(short, long, default_value_t = 20)]
        games: u32,
        /// MCTS simulations per move for both agents
        #[arg(short = 's', long, default_value_t = 100)]
        sims: u32,
        /// Maximum half-moves before a game is declared a draw
        #[arg(long, default_value_t = 300)]
        max_moves: usize,
        /// Network channel width (8 for quick smoke test; 256 for full AlphaZero)
        #[arg(long, default_value_t = 8)]
        channels: i64,
        /// Number of residual blocks (2 for quick; 20 for full AlphaZero)
        #[arg(long, default_value_t = 2)]
        blocks: usize,
        /// Optional .safetensors checkpoint to load into the network
        #[arg(short, long)]
        checkpoint: Option<std::path::PathBuf>,
        /// Print a result line for every game
        #[arg(short, long)]
        verbose: bool,
        /// RNG seed for reproducibility
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.command.unwrap_or(Commands::Usi) {
        Commands::Usi => run_usi_loop(),

        Commands::Perft { depth, divide } => {
            let mut board = Board::startpos();
            if divide {
                perft_divide(&mut board, depth);
            } else {
                for d in 1..=depth {
                    let count = perft(&mut board, d);
                    println!("Perft({d}) = {count}");
                }
            }
        }

        Commands::Startpos => {
            println!("{}", Board::startpos().to_sfen());
        }

        Commands::Train {
            channels,
            blocks,
            workers,
            checkpoint_dir,
            checkpoint_every,
            total_steps,
            pit_games,
            resume,
        } => {
            use std::sync::Mutex;
            use rand::SeedableRng;
            use rand::rngs::StdRng;
            use shogi_core::nn;
            use shogi_core::orchestrate::{OrchestrationConfig, run_loop};
            use shogi_core::replay_buffer::ReplayBuffer;
            use shogi_core::selfplay::SelfPlayConfig;
            use shogi_core::train::{TrainConfig, Trainer};
            use shogi_core::worker::WorkerPool;

            let device = nn::device();

            let mut trainer = Trainer::new(device, channels, blocks, TrainConfig::default());

            if let Some(ref path) = resume {
                trainer.resume(path);
            }

            let buffer = Arc::new(Mutex::new(ReplayBuffer::new(1_000_000)));

            let shutdown = Arc::new(AtomicBool::new(false));
            let sd = Arc::clone(&shutdown);
            ctrlc::set_handler(move || {
                println!("\nShutdown signal received — finishing current game…");
                sd.store(true, Ordering::Relaxed);
            })
            .expect("failed to set Ctrl-C handler");

            let pool = if workers > 0 {
                Some(WorkerPool::spawn(
                    workers,
                    &trainer.vs,
                    channels,
                    blocks,
                    SelfPlayConfig::default(),
                    Arc::clone(&buffer),
                    42,
                ))
            } else {
                None
            };

            let config = OrchestrationConfig {
                steps_per_broadcast: 100,
                total_steps,
                fill_poll_ms: 200,
                checkpoint_every,
                checkpoint_dir,
                pit_games,
            };

            let mut rng = StdRng::seed_from_u64(0);
            run_loop(&mut trainer, buffer, pool.as_ref(), &config, &mut rng, shutdown);

            if let Some(p) = pool {
                p.join();
            }
            println!("Training complete at step {}.", trainer.step);
        }

        Commands::Bench {
            games,
            sims,
            max_moves,
            channels,
            blocks,
            checkpoint,
            verbose,
            seed,
        } => {
            use shogi_core::bench::{AgentKind, run_match};
            use shogi_core::nn::{self, checkpoint as ckpt};

            let device = nn::device();

            let (mut vs, net) = ckpt::build_with_config(device, channels, blocks);

            if let Some(ref path) = checkpoint {
                ckpt::load(&mut vs, path)
                    .unwrap_or_else(|e| panic!("failed to load checkpoint {:?}: {e}", path));
                println!("Loaded checkpoint: {:?}", path);
            } else {
                println!(
                    "No checkpoint — network has random weights ({channels}ch, {blocks} blocks)."
                );
            }

            println!(
                "Match: Net vs Rollout  |  {games} games  |  {sims} sims/move  |  max {max_moves} half-moves"
            );

            let net_agent = AgentKind::Network { net: &net, device };
            let rollout_agent = AgentKind::Rollout;

            let start = std::time::Instant::now();

            // Run without gradient tracking for inference speed.
            let result = tch::no_grad(|| {
                run_match(
                    &net_agent, "Net",
                    &rollout_agent, "Rollout",
                    games, sims, max_moves,
                    verbose, seed,
                )
            });

            result.print_summary("Net", "Rollout", start.elapsed());
        }
    }
}
