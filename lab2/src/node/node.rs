use crate::board::board::{Board, GameStatus};
use crate::board::token::TokenColor;
use serde::{Deserialize, Serialize};

const WIN_VALUE: f64 = 1.0;
const LOSE_VALUE: f64 = -1.0;
const NEUTRAL_VALUE: f64 = 0.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    color: TokenColor,
    pub column: usize,
    pub value: Option<f64>,
    pub children: Vec<Node>,
    status: GameStatus,
}

impl Node {
    pub fn new(color: TokenColor, column: usize, status: GameStatus) -> Node {
        return Self {
            color,
            column,
            status,
            value: None,
            children: vec![],
        };
    }

    pub fn add_child(self: &mut Self, node: Node) {
        self.children.push(node)
    }

    pub fn build_tree(self: &mut Self, board: &mut Board, max_depth: usize, curr_depth: usize) {
        if curr_depth > max_depth {
            return;
        }

        for i in 0..board.columns {
            if !board.is_move_legal(i) {
                continue;
            }

            board.make_move(i, self.color.invert()).unwrap();
            match board.get_status() {
                GameStatus::Finished(status) => {
                    board.undo_move(i).unwrap();

                    self.add_child(Node::new(
                        self.color.invert(),
                        i,
                        GameStatus::Finished(status),
                    ));

                    continue;
                }

                GameStatus::InProgress => {
                    let mut child = Node::new(self.color.invert(), i, GameStatus::InProgress);

                    child.build_tree(board, max_depth, curr_depth + 1);

                    board.undo_move(i).unwrap();

                    self.add_child(child);
                }
            }
        }
    }

    pub fn calculate_value(
        self: &mut Self,
        board: &mut Board,
        cpu_color: TokenColor,
        player_color: TokenColor,
    ) {
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
                board.make_move(child.column, child.color).unwrap();
                child.calculate_value(board, cpu_color, player_color);
                board.undo_move(child.column).unwrap();
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
