use super::{assignment::Assignment, tag::Tag};
use crate::{
    board::board::{Board, GameStatus},
    board::token::TokenColor,
    node::node::Node,
};

use mpi::{topology::*, traits::*};
use std::{
    io::{self, Write},
    time::Instant,
};

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
        loop {
            let input_column = self.player_input();

            let status = match self.board.get_status() {
                GameStatus::Finished(color) => {
                    println!("The winner is {:#?}!", color);
                    break;
                }
                _ => GameStatus::InProgress,
            };

            let start = Instant::now();

            let mut root: Node = Node::new(self.player_color, input_column, status);
            root.build_tree(&mut self.board, 1, 0);

            let mut requests: Vec<Assignment> = vec![];
            let mut responses: Vec<Assignment> = vec![];

            self.create_requests(&root, &mut requests);

            self.send_requests(&mut requests, &mut responses);

            let best_move = self.calculate_best_move(&mut root, responses);

            self.board.make_move(best_move, self.cpu_color).unwrap();

            println!(
                "{}",
                root.children
                    .iter()
                    .map(|x| x.value.unwrap().to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            );
            println!("Best move: {}", best_move);
            println!("Elapsed time: {:#?}", start.elapsed());

            self.board.show();

            match self.board.get_status() {
                GameStatus::Finished(color) => {
                    println!("The winner is {:#?}!", color);
                    break;
                }
                _ => GameStatus::InProgress,
            };
        }

        self.notify_game_finished();
    }

    fn player_input(&mut self) -> usize {
        let mut input_line: String;
        let mut input_column: usize;

        loop {
            print!("Enter column index: ");
            io::stdout().flush().unwrap();
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

            let Ok(_) = self.board.make_move(input_column, self.player_color) else {
                println!("Invalid input");
                continue;
            };

            self.board.show();

            return input_column;
        }
    }

    fn create_requests(&mut self, root: &Node, tasks: &mut Vec<Assignment>) {
        for (i, child1) in root.children.iter().enumerate() {
            for (j, child2) in child1.children.iter().enumerate() {
                let mut board_clone: Board = self.board.clone();

                board_clone
                    .make_move(child1.column, self.cpu_color)
                    .unwrap();

                board_clone
                    .make_move(child2.column, self.player_color)
                    .unwrap();

                let task: Assignment = Assignment::new(child2.clone(), board_clone, (i, j));

                tasks.push(task);
            }
        }
    }

    fn send_requests(&mut self, tasks: &mut Vec<Assignment>, results: &mut Vec<Assignment>) {
        let mut rank: i32 = 0;
        let task_count: usize = tasks.len();

        loop {
            if task_count == 0 {
                break;
            }

            if rank == self.world.rank() {
                rank += 1;
                continue;
            }

            if rank == self.world.size() {
                rank = 0;
                continue;
            }

            let (msg, status) = self.world.process_at_rank(rank).receive_vec::<u8>();

            if status.tag() == Tag::Request as i32 {
                let task: Assignment = tasks.pop().unwrap();

                self.world
                    .process_at_rank(rank)
                    .send_with_tag(&bincode::serialize(&task).unwrap(), Tag::Response as i32);

                if tasks.len() == 0 {
                    rank = 0;
                    continue;
                }
            } else {
                results.push(bincode::deserialize(&msg).unwrap());

                if results.len() == task_count {
                    break;
                }
            }

            rank += 1;
        }
    }

    fn calculate_best_move(&mut self, root: &mut Node, responses: Vec<Assignment>) -> usize {
        for result in responses {
            let (i, j) = result.indexes;
            root.children[i].children[j] = result.node;
        }

        let mut best_move: Option<usize> = None;
        let mut max_value: Option<f64> = None;

        for child in root.children.iter_mut() {
            child.calculate_value(&mut self.board, self.cpu_color, self.player_color);

            if best_move == None || child.value > max_value {
                best_move = Some(child.column);
                max_value = child.value;
            }
        }

        return best_move.unwrap();
    }

    fn notify_game_finished(&mut self) {
        for rank in 0..self.world.size() {
            if rank == self.world.rank() {
                continue;
            }

            self.world
                .process_at_rank(rank)
                .receive_vec_with_tag::<u8>(Tag::Request as i32);

            self.world
                .process_at_rank(rank)
                .send_with_tag(&0, Tag::Finished as i32);
        }
    }
}
