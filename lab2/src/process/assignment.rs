use crate::{board::board::Board, node::node::Node};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct Assignment {
    pub node: Node,
    pub board: Board,
    pub indexes: (usize, usize),
}

impl Assignment {
    pub fn new(node: Node, board: Board, indexes: (usize, usize)) -> Assignment {
        return Self {
            node,
            board,
            indexes,
        };
    }
}
