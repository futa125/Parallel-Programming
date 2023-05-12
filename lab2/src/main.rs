mod board;
use std::io;

fn main() {
    let stdin = io::stdin();
    let mut buf = String::new();
    let mut column: usize;
    let mut color: board::TokenColor = board::TokenColor::Yellow;

    let mut board = board::Board::default();

    loop {
        match stdin.read_line(&mut buf) {
            Err(err) => println!("{err}"),
            _ => (),
        }

        match buf.trim().parse::<usize>() {
            Err(err) => {
                println!("{err}");
                buf.clear();
                continue;
            }
            Ok(v) => column = v,
        }

        match board.make_move(column, color) {
            Ok(status) => match status {
                board::GameStatus::Finished(winner) => {
                    println!("Winner is: {:?}", winner);
                    board.show();
                    break;
                }
                _ => (),
            },
            Err(err) => {
                println!("{err}");
                buf.clear();
                continue;
            }
        }

        board.show();
        buf.clear();

        if color == board::TokenColor::Yellow {
            color = board::TokenColor::Red
        } else {
            color = board::TokenColor::Yellow
        }
    }
}
