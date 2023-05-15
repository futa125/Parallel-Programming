use crate::{node::node::Node, board::board::Board};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct Task {
    pub node: Node,
    pub board: Board,
    pub indexes: Vec<usize>,
}

impl Task {
    pub fn new(node: Node, board: Board) -> Task {
        return Self {
            node,
            board,
            indexes: vec![],
        }
    }
}
