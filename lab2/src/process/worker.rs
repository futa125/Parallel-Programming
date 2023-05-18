use super::assignment::Assignment;
use crate::{board::token::TokenColor, process::tag::Tag};
use mpi::{topology::*, traits::*};

pub struct Worker {
    world: SystemCommunicator,
    master_rank: i32,
    cpu_color: TokenColor,
    player_color: TokenColor,
    depth: usize,
}

impl Worker {
    pub fn new(world: SystemCommunicator, master_rank: i32, depth: usize) -> Self {
        return Self {
            world,
            master_rank,
            cpu_color: TokenColor::Red,
            player_color: TokenColor::Yellow,
            depth,
        };
    }

    pub fn run(self: &Self) {
        loop {
            self.world
                .process_at_rank(self.master_rank)
                .send_with_tag::<u8>(&0, Tag::Request as i32);

            let (msg, status) = self
                .world
                .process_at_rank(self.master_rank)
                .receive_vec::<u8>();

            if status.tag() == Tag::Finished as i32 {
                break;
            }

            if status.tag() == Tag::Response as i32 {
                let mut assignment: Assignment = bincode::deserialize(&msg).unwrap();

                assignment
                    .node
                    .build_tree(&mut assignment.board, self.depth, 0);

                assignment.node.calculate_value(
                    &mut assignment.board,
                    self.cpu_color,
                    self.player_color,
                );

                let task_encoded: Vec<u8> = bincode::serialize(&assignment).unwrap();

                self.world
                    .process_at_rank(self.master_rank)
                    .send_with_tag(&task_encoded, Tag::Result as i32);
            }
        }
    }
}
