extern crate core_affinity;

mod board;
mod node;
mod process;

use board::board::Board;
use mpi::traits::*;
use process::master::Master;
use process::worker::Worker;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
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
