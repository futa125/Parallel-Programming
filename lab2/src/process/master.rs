extern crate bincode;
extern crate mpi;
extern crate serde_json;

use crate::{
    board::board::{Board, GameStatus, TokenColor},
    node::node::Node,
    process::tag::Tag,
};
use mpi::{topology::*, traits::*};
use std::time::Instant;
use std::{
    io::{self, stdout, Write},
    thread::sleep,
    time::Duration,
};

use super::task::Task;

pub struct Master {
    world: SystemCommunicator,
    board: Board,
    cpu_color: TokenColor,
    player_color: TokenColor,
}

impl Master {
    pub fn new(world: SystemCommunicator, board: Board) -> Self {
        return Self {
            world,
            board,
            cpu_color: TokenColor::Red,
            player_color: TokenColor::Yellow,
        };
    }

    pub fn run(self: &mut Self) {
        let mut input_line: String;
        let mut input_column: usize;
        let mut root: Node;
        let mut game_status: GameStatus;
        let mut tasks: Vec<Task>;
        let mut results: Vec<Task>;
        let mut task_count: usize;

        loop {
            print!("Enter column index: ");
            _ = io::stdout().flush();
            input_line = String::new();

            let Ok(_) = io::stdin().read_line(&mut input_line) else {
                println!("Invalid input");
                continue;
            };

            input_column = match input_line.trim().parse() {
                Ok(x) => x,
                Err(_) => {
                    println!("Invalid input");
                    continue;
                }
            };

            match self.board.make_move(input_column, self.player_color) {
                Ok(status) => match status {
                    GameStatus::Finished(winner) => {
                        self.board.show();
                        println!("Winner is {:#?}", winner);
                        break;
                    }
                    GameStatus::InProgress => game_status = GameStatus::InProgress,
                },
                Err(_) => {
                    println!("Illegal move");
                    continue;
                }
            }

            self.board.show();

            root = Node::new(
                self.board.clone(),
                game_status,
                self.player_color,
                input_column,
            );
            root.build_tree(1, 0);

            tasks = vec![];
            results = vec![];

            for i in 0..root.children.len() {
                for j in 0..root.children[i].children.len() {
                    let mut task = Task::new(root.children[i].children[j].clone());
                    task.indexes.push(i);
                    task.indexes.push(j);

                    tasks.push(task);
                }
            }

            task_count = tasks.len();
            let start = Instant::now();
            loop {
                let (msg, status) = self.world.any_process().receive_vec::<u8>();

                if status.tag() == Tag::RequestWork as i32 {
                    let Some(task) = tasks.pop() else {
                        continue;
                    };

                    self.world
                        .process_at_rank(status.source_rank())
                        .send_with_tag(&bincode::serialize(&task).unwrap(), Tag::Assignment as i32);
                } else if status.tag() == Tag::Result as i32 {
                    results.push(bincode::deserialize(&msg).unwrap());

                    if results.len() == task_count {
                        break;
                    }
                }
            }

            println!("{:#?}", start.elapsed());

            for result in results {
                let (i, j) = (result.indexes[0], result.indexes[1]);
                root.children[i].children[j] = result.node;
            }
            
            let mut opt_move: Option<usize> = None;
            let mut max_value: Option<f64> = None;
            for mut child in root.children {
                child.calculate_value(self.cpu_color, self.player_color);
                print!("{:#?}, ", child.value.unwrap());
                if max_value == None || child.value > max_value {
                    max_value = child.value;
                    opt_move = Some(child.column);
                }
            }
            println!();
            
            match self.board.make_move(opt_move.unwrap(), self.cpu_color) {
                Ok(status) => match status {
                    GameStatus::Finished(winner) => {
                        self.board.show();
                        println!("Winner is {:#?}", winner);
                        break;
                    }
                    GameStatus::InProgress => (),
                },
                Err(_) => {
                    println!("Illegal move");
                    continue;
                }
            }
            self.board.show();
        }
    }
}
