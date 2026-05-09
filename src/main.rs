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
