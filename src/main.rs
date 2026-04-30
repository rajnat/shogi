use clap::{Parser, Subcommand};
use shogi_core::board::Board;
use shogi_core::perft::{perft, perft_divide};

#[derive(Parser)]
#[command(name = "shogi", version, about = "Shogi engine")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run perft from startpos
    Perft {
        #[arg(short, long, default_value_t = 5)]
        depth: u32,
        #[arg(short = 'D', long)]
        divide: bool,
    },
    /// Show startpos SFEN
    Startpos,
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Perft { depth, divide } => {
            let mut board = Board::startpos();
            if divide {
                perft_divide(&mut board, depth);
            } else {
                for d in 1..=depth {
                    let count = perft(&mut board, d);
                    println!("Perft({}) = {}", d, count);
                }
            }
        }
        Commands::Startpos => {
            let board = Board::startpos();
            println!("{}", board.to_sfen());
        }
    }
}
