extern crate mpi;

mod board;
mod node;
mod process;

use board::board::GameStatus;
use process::worker::Worker;
use std::io;

use crate::board::board::{Board, TokenColor};
use crate::node::node::Node;
use crate::process::master::Master;
use mpi::traits::*;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size: i32 = world.size();
    let rank: i32 = world.rank();
    let master_rank: i32 = 0;

    let board = Board::default();

    let mut master = Master::new(world, board);
    let worker: Worker = Worker::new(world, master_rank);

    if rank == master_rank {
        master.run();
    } else {
        worker.run();
    }
}
