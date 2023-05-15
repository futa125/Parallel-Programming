use crate::node::node::Node;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct Task {
    pub node: Node,
    pub indexes: Vec<usize>,
}

impl Task {
    pub fn new(node: Node) -> Task {
        return Self {
            node,
            indexes: vec![],
        }
    }
}
