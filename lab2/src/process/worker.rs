use super::task::Task;
use crate::{board::token::TokenColor, process::tag::Tag};
use mpi::{topology::*, traits::*};

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
                .send_with_tag::<u8>(&0, Tag::RequestWork as i32);

            let (assignment, status) = self
                .world
                .process_at_rank(self.master_rank)
                .receive_vec::<u8>();

            if status.tag() == Tag::Finished as i32 {
                break;
            } else if status.tag() == Tag::Assignment as i32 {
                let mut task: Task = bincode::deserialize(&assignment).unwrap();

                task.node.build_tree(&mut task.board, 6, 2);

                task.node
                    .calculate_value(&mut task.board, self.cpu_color, self.player_color);

                let task_encoded: Vec<u8> = bincode::serialize(&task).unwrap();

                self.world
                    .process_at_rank(self.master_rank)
                    .send_with_tag(&task_encoded, Tag::Result as i32);
            }
        }
    }
}
