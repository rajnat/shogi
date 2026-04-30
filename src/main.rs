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
    }
}
