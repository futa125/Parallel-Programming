extern crate bincode;
extern crate mpi;
extern crate serde_json;

use crate::{
    board::board::{Board, GameStatus, TokenColor},
    node::node::Node,
    process::tag::Tag,
};
use mpi::{topology::*, traits::*};
use std::{
    io::{self, Write},
    thread::sleep,
    time::Duration,
};
use super::task::Task;

pub struct Worker {
    world: SystemCommunicator,
    master_rank: i32,
    cpu_color: TokenColor,
    player_color: TokenColor,
}

impl Worker {
    pub fn new(world: SystemCommunicator, master_rank: i32) -> Self {
        return Self {
            world,
            master_rank,
            cpu_color: TokenColor::Red,
            player_color: TokenColor::Yellow,
        };
    }

    pub fn run(self: &Self) {
        loop {
            self.world
                .process_at_rank(self.master_rank)
                .send_with_tag(&0, Tag::RequestWork as i32);

            let (assignment, _) = self
                .world
                .process_at_rank(self.master_rank)
                .receive_vec_with_tag::<u8>(Tag::Assignment as i32);


            let mut task: Task = bincode::deserialize(&assignment).unwrap();

            task.node.build_tree(6, 2);
            
            task.node.calculate_value(self.cpu_color, self.player_color);

            let task_encoded: Vec<u8> = bincode::serialize(&task).unwrap();

            self.world
                .process_at_rank(self.master_rank)
                .send_with_tag(&task_encoded, Tag::Result as i32);
        }
    }
}
