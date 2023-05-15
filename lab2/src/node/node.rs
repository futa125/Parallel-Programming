extern crate serde;

use crate::board::board::{Board, GameStatus, TokenColor};
use serde::{Serialize, Deserialize};

const WIN_VALUE: f64 = 1.0;
const LOSE_VALUE: f64 = -1.0;
const NEUTRAL_VALUE: f64 = 0.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub board: Board,
    pub status: GameStatus,
    pub color: TokenColor,
    pub column: usize,
    pub value: Option<f64>,
    pub children: Vec<Node>,
}

impl Node {
    pub fn new(board: Board, status: GameStatus, color: TokenColor, column: usize) -> Node {
        return Self {
            board,
            status,
            color,
            column,
            value: None,
            children: vec![],
        };
    }

    pub fn add_child(self: &mut Self, node: Node) {
        self.children.push(node)
    }

    pub fn build_tree(self: &mut Self, max_depth: usize, curr_depth: usize) {
        if curr_depth > max_depth {
            return;
        }

        for i in 0..self.board.columns {
            let mut board_clone = self.board.clone();

            let Ok(game_status) = board_clone.make_move(i, self.color.invert()) else {
                continue;
            };

            let mut child = Node::new(board_clone, game_status, self.color.invert(), i);

            child.build_tree(max_depth, curr_depth + 1);

            self.add_child(child);
        }
    }

    pub fn calculate_value(self: &mut Self, cpu_color: TokenColor, player_color: TokenColor) {
        if self.children.len() == 0 {
            match self.status {
                GameStatus::Finished(color) => {
                    if color == cpu_color {
                        self.value = Some(WIN_VALUE);
                    } else {
                        self.value = Some(LOSE_VALUE);
                    }
                }
                GameStatus::InProgress => self.value = Some(NEUTRAL_VALUE),
            }

            return;
        }

        for child in self.children.as_mut_slice() {
            if child.value.is_none() {
                child.calculate_value(cpu_color, player_color);
            }
        }

        if self
            .children
            .iter()
            .any(|x| x.value.unwrap_or(0.0) == WIN_VALUE && x.color == cpu_color)
        {
            self.value = Some(WIN_VALUE);
        } else if self
            .children
            .iter()
            .any(|x| x.value.unwrap_or(0.0) == LOSE_VALUE && x.color == player_color)
        {
            self.value = Some(LOSE_VALUE);
        } else if self
            .children
            .iter()
            .all(|x| x.value.unwrap_or(0.0) == WIN_VALUE)
        {
            self.value = Some(WIN_VALUE);
        } else if self
            .children
            .iter()
            .all(|x| x.value.unwrap_or(0.0) == LOSE_VALUE)
        {
            self.value = Some(LOSE_VALUE);
        } else {
            self.value = Some(
                self.children
                    .iter()
                    .map(|x| x.value.unwrap_or(0.0))
                    .sum::<f64>()
                    / (self.children.len() as f64),
            )
        }

        return;
    }
}
