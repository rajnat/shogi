use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

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
        /// Directory to write run outputs into
        #[arg(long, default_value = "runs/default")]
        run_dir: String,
        /// Path to write training metrics JSONL (defaults to <run-dir>/metrics.jsonl)
        #[arg(long)]
        metrics_jsonl: Option<PathBuf>,
        /// Path to write evaluation metrics JSONL (defaults to <run-dir>/eval.jsonl)
        #[arg(long)]
        eval_jsonl: Option<PathBuf>,
        /// Directory to write checkpoint files into
        #[arg(long, default_value = "checkpoints")]
        checkpoint_dir: String,
        /// Save a checkpoint every N training steps (0 = never)
        #[arg(long, default_value_t = 1000)]
        checkpoint_every: u64,
        /// Stop after N total training steps (0 = run until Ctrl-C)
        #[arg(long, default_value_t = 0)]
        total_steps: u64,
        /// Mini-batch size sampled from the replay buffer each training step
        #[arg(long, default_value_t = 512)]
        batch_size: usize,
        /// Adam optimizer learning rate
        #[arg(long, default_value_t = 1e-3)]
        learning_rate: f64,
        /// Adam optimizer weight decay
        #[arg(long, default_value_t = 1e-4)]
        weight_decay: f64,
        /// Minimum replay-buffer positions required before training starts
        #[arg(long, default_value_t = 10_000)]
        min_buffer_size: usize,
        /// Log training losses every N steps
        #[arg(long, default_value_t = 100)]
        log_every: u64,
        /// MCTS simulations per self-play move
        #[arg(long, default_value_t = 800)]
        sims: u32,
        /// Move-selection temperature before the temperature drop ply
        #[arg(long, default_value_t = 1.0)]
        temperature_high: f32,
        /// Move-selection temperature after the temperature drop ply
        #[arg(long, default_value_t = 0.0)]
        temperature_low: f32,
        /// Half-move ply at which self-play temperature drops
        #[arg(long, default_value_t = 30)]
        temperature_drop_ply: u32,
        /// Dirichlet alpha for root exploration noise
        #[arg(long, default_value_t = 0.15)]
        dirichlet_alpha: f32,
        /// Fraction of Dirichlet noise mixed into root priors
        #[arg(long, default_value_t = 0.25)]
        dirichlet_epsilon: f32,
        /// PUCT exploration constant
        #[arg(long, default_value_t = 1.0)]
        c_puct: f32,
        /// Resign when root value estimate stays below this threshold
        #[arg(long, default_value_t = -0.9, allow_hyphen_values = true)]
        resign_threshold: f32,
        /// Do not allow resignations before this half-move ply
        #[arg(long, default_value_t = 30)]
        resign_min_ply: u32,
        /// Consecutive below-threshold plies required before resigning
        #[arg(long, default_value_t = 5)]
        resign_consecutive: u32,
        /// Maximum half-moves before a self-play game is declared drawn
        #[arg(long, default_value_t = 512)]
        max_moves: usize,
        /// Leaf positions batched per neural-net MCTS evaluation
        #[arg(long, default_value_t = 8)]
        mcts_batch_size: usize,
        /// RNG seed for training and self-play workers
        #[arg(long, default_value_t = 42)]
        seed: u64,
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn train_cli_parses_training_hyperparameters() {
        let cli = Cli::try_parse_from([
            "shogi",
            "train",
            "--batch-size",
            "128",
            "--learning-rate",
            "0.0003",
            "--weight-decay",
            "0.00001",
            "--min-buffer-size",
            "2048",
            "--log-every",
            "25",
            "--seed",
            "1234",
        ])
        .expect("train CLI should parse training hyperparameters");

        let Some(Commands::Train {
            batch_size,
            learning_rate,
            weight_decay,
            min_buffer_size,
            log_every,
            seed,
            ..
        }) = cli.command
        else {
            panic!("expected train command");
        };

        assert_eq!(batch_size, 128);
        assert!((learning_rate - 0.0003).abs() < f64::EPSILON);
        assert!((weight_decay - 0.00001).abs() < f64::EPSILON);
        assert_eq!(min_buffer_size, 2048);
        assert_eq!(log_every, 25);
        assert_eq!(seed, 1234);
    }

    #[test]
    fn train_cli_resolves_default_run_output_paths() {
        let cli = Cli::try_parse_from(["shogi", "train", "--run-dir", "runs/example"])
            .expect("train CLI should parse run dir");

        let Some(Commands::Train {
            run_dir,
            metrics_jsonl,
            eval_jsonl,
            ..
        }) = cli.command
        else {
            panic!("expected train command");
        };

        let (metrics_path, eval_path) =
            resolve_run_output_paths(&run_dir, metrics_jsonl, eval_jsonl);

        assert_eq!(metrics_path, PathBuf::from("runs/example/metrics.jsonl"));
        assert_eq!(eval_path, PathBuf::from("runs/example/eval.jsonl"));
    }

    #[test]
    fn train_cli_respects_explicit_run_output_paths() {
        let cli = Cli::try_parse_from([
            "shogi",
            "train",
            "--run-dir",
            "runs/example",
            "--metrics-jsonl",
            "custom/metrics.jsonl",
            "--eval-jsonl",
            "custom/eval.jsonl",
        ])
        .expect("train CLI should parse explicit run output paths");

        let Some(Commands::Train {
            run_dir,
            metrics_jsonl,
            eval_jsonl,
            ..
        }) = cli.command
        else {
            panic!("expected train command");
        };

        let (metrics_path, eval_path) =
            resolve_run_output_paths(&run_dir, metrics_jsonl, eval_jsonl);

        assert_eq!(metrics_path, PathBuf::from("custom/metrics.jsonl"));
        assert_eq!(eval_path, PathBuf::from("custom/eval.jsonl"));
    }

    #[test]
    fn train_cli_parses_selfplay_hyperparameters() {
        let cli = Cli::try_parse_from([
            "shogi",
            "train",
            "--sims",
            "64",
            "--temperature-high",
            "1.5",
            "--temperature-low",
            "0.1",
            "--temperature-drop-ply",
            "20",
            "--dirichlet-alpha",
            "0.3",
            "--dirichlet-epsilon",
            "0.4",
            "--c-puct",
            "2.5",
            "--resign-threshold",
            "-0.8",
            "--resign-min-ply",
            "12",
            "--resign-consecutive",
            "3",
            "--max-moves",
            "128",
            "--mcts-batch-size",
            "16",
        ])
        .expect("train CLI should parse self-play hyperparameters");

        let Some(Commands::Train {
            sims,
            temperature_high,
            temperature_low,
            temperature_drop_ply,
            dirichlet_alpha,
            dirichlet_epsilon,
            c_puct,
            resign_threshold,
            resign_min_ply,
            resign_consecutive,
            max_moves,
            mcts_batch_size,
            ..
        }) = cli.command
        else {
            panic!("expected train command");
        };

        assert_eq!(sims, 64);
        assert!((temperature_high - 1.5).abs() < f32::EPSILON);
        assert!((temperature_low - 0.1).abs() < f32::EPSILON);
        assert_eq!(temperature_drop_ply, 20);
        assert!((dirichlet_alpha - 0.3).abs() < f32::EPSILON);
        assert!((dirichlet_epsilon - 0.4).abs() < f32::EPSILON);
        assert!((c_puct - 2.5).abs() < f32::EPSILON);
        assert!((resign_threshold - -0.8).abs() < f32::EPSILON);
        assert_eq!(resign_min_ply, 12);
        assert_eq!(resign_consecutive, 3);
        assert_eq!(max_moves, 128);
        assert_eq!(mcts_batch_size, 16);
    }
}

fn resolve_run_output_paths(
    run_dir: &str,
    metrics_jsonl: Option<PathBuf>,
    eval_jsonl: Option<PathBuf>,
) -> (PathBuf, PathBuf) {
    let run_dir = PathBuf::from(run_dir);
    let metrics_path = metrics_jsonl.unwrap_or_else(|| run_dir.join("metrics.jsonl"));
    let eval_path = eval_jsonl.unwrap_or_else(|| run_dir.join("eval.jsonl"));
    (metrics_path, eval_path)
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
            run_dir,
            metrics_jsonl,
            eval_jsonl,
            checkpoint_dir,
            checkpoint_every,
            total_steps,
            batch_size,
            learning_rate,
            weight_decay,
            min_buffer_size,
            log_every,
            sims,
            temperature_high,
            temperature_low,
            temperature_drop_ply,
            dirichlet_alpha,
            dirichlet_epsilon,
            c_puct,
            resign_threshold,
            resign_min_ply,
            resign_consecutive,
            max_moves,
            mcts_batch_size,
            seed,
            pit_games,
            resume,
        } => {
            use rand::rngs::StdRng;
            use rand::SeedableRng;
            use shogi_core::metrics::JsonlWriter;
            use shogi_core::nn;
            use shogi_core::orchestrate::{run_loop, OrchestrationConfig};
            use shogi_core::replay_buffer::ReplayBuffer;
            use shogi_core::selfplay::SelfPlayConfig;
            use shogi_core::train::{TrainConfig, Trainer};
            use shogi_core::worker::WorkerPool;
            use std::sync::Mutex;

            let device = nn::device();
            let (metrics_jsonl_path, eval_jsonl_path) =
                resolve_run_output_paths(&run_dir, metrics_jsonl, eval_jsonl);
            std::fs::create_dir_all(&run_dir).expect("failed to create run dir");

            {
                use shogi_core::metrics::{RunConfig, write_run_config};
                let cfg = RunConfig {
                    version: env!("CARGO_PKG_VERSION").to_string(),
                    unix_timestamp: RunConfig::unix_now(),
                    channels,
                    blocks,
                    workers,
                    seed,
                    run_dir: run_dir.clone(),
                    metrics_jsonl: metrics_jsonl_path.display().to_string(),
                    eval_jsonl: eval_jsonl_path.display().to_string(),
                    checkpoint_dir: checkpoint_dir.clone(),
                    batch_size,
                    learning_rate,
                    weight_decay,
                    min_buffer_size,
                    log_every,
                    checkpoint_every,
                    total_steps,
                    pit_games,
                    sims,
                    temperature_high,
                    temperature_low,
                    temperature_drop_ply,
                    dirichlet_alpha,
                    dirichlet_epsilon,
                    c_puct,
                    resign_threshold,
                    resign_min_ply,
                    resign_consecutive,
                    max_moves,
                    mcts_batch_size,
                    resume: resume.as_ref().map(|p| p.display().to_string()),
                };
                write_run_config(
                    &std::path::PathBuf::from(&run_dir).join("run_config.json"),
                    &cfg,
                )
                .expect("failed to write run_config.json");
            }

            let mut metrics_writer = Some(
                JsonlWriter::new(&metrics_jsonl_path)
                    .expect("failed to create metrics JSONL writer"),
            );
            let mut eval_writer = Some(
                JsonlWriter::new(&eval_jsonl_path)
                    .expect("failed to create eval JSONL writer"),
            );

            let train_config = TrainConfig {
                batch_size,
                learning_rate,
                weight_decay,
                min_buffer_size,
                log_every,
            };
            let mut trainer = Trainer::new(device, channels, blocks, train_config);

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

            let selfplay_config = SelfPlayConfig {
                num_simulations: sims,
                temperature_high,
                temperature_drop_ply,
                temperature_low,
                resign_threshold,
                resign_min_ply,
                resign_consecutive,
                max_moves,
                dirichlet_alpha,
                dirichlet_epsilon,
                c_puct,
                mcts_batch_size,
            };

            let pool = if workers > 0 {
                Some(WorkerPool::spawn(
                    workers,
                    &trainer.vs,
                    channels,
                    blocks,
                    selfplay_config,
                    Arc::clone(&buffer),
                    seed,
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

            let mut rng = StdRng::seed_from_u64(seed);
            run_loop(
                &mut trainer,
                buffer,
                pool.as_ref(),
                &config,
                &mut rng,
                shutdown,
                metrics_writer.as_mut(),
                eval_writer.as_mut(),
            );

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
            use shogi_core::bench::{run_match, AgentKind};
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
                    &net_agent,
                    "Net",
                    &rollout_agent,
                    "Rollout",
                    games,
                    sims,
                    max_moves,
                    verbose,
                    seed,
                )
            });

            result.print_summary("Net", "Rollout", start.elapsed());
        }
    }
}
