use super::{tag::Tag, task::Task};
use crate::{
    board::board::{Board, GameStatus},
    board::token::TokenColor,
    node::node::Node,
};
use mpi::{topology::*, traits::*};
use std::{
    io::{self, Write},
    thread::sleep,
    time::Duration,
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
        let mut root: Node;

        let mut input_column: usize;
        let mut status: GameStatus;

        let mut tasks: Vec<Task>;
        let mut results: Vec<Task>;
        let mut task_count: usize;

        loop {
            input_column = self.player_input();

            status = match self.board.get_status() {
                GameStatus::Finished(color) => {
                    println!("The winner is {:#?}!", color);
                    break;
                }
                _ => GameStatus::InProgress,
            };

            root = Node::new(self.player_color, input_column, status);

            root.build_tree(&mut self.board, 1, 0);

            tasks = vec![];
            results = vec![];

            for i in 0..root.children.len() {
                for j in 0..root.children[i].children.len() {
                    let mut board_clone = self.board.clone();
                    if !board_clone.is_move_legal(i) {
                        continue;
                    }
                    board_clone.make_move(i, self.cpu_color).unwrap();
                    if !board_clone.is_move_legal(j) {
                        continue;
                    }
                    board_clone.make_move(j, self.player_color).unwrap();
                    let mut task = Task::new(root.children[i].children[j].clone(), board_clone);
                    task.indexes.push(i);
                    task.indexes.push(j);

                    tasks.push(task);
                }
            }

            task_count = tasks.len();
            let start = Instant::now();
            loop {
                let (msg, status) = self.world.any_process().receive_vec::<u8>();
                println!("{}", status.source_rank());

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
                child.calculate_value(&mut self.board, self.cpu_color, self.player_color);
                print!("{:#?}, ", child.value.unwrap());
                if max_value == None || child.value > max_value {
                    max_value = child.value;
                    opt_move = Some(child.column);
                }
            }
            println!();

            self.board
                .make_move(opt_move.unwrap(), self.cpu_color)
                .unwrap();
            self.board.show();

            match self.board.get_status() {
                GameStatus::Finished(color) => {
                    println!("The winner is {:#?}!", color);
                    break;
                }
                _ => GameStatus::InProgress,
            };
        }

        for rank in 0..self.world.size() {
            if rank == self.world.rank() {
                continue;
            }

            self.world
                .process_at_rank(rank)
                .send_with_tag(&0, Tag::Finished as i32);
        }
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
}
