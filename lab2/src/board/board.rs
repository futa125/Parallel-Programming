use super::token::TokenColor;
use serde::{Deserialize, Serialize};
use std::fmt;

const DEFAULT_ROWS: usize = 6;
const DEFAULT_COLUMNS: usize = 7;

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
enum Field {
    Empty,
    Token(TokenColor),
}

impl fmt::Display for Field {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            Field::Empty => write!(f, "⚪"),
            Field::Token(color) => match color {
                TokenColor::Red => write!(f, "🔴"),
                TokenColor::Yellow => write!(f, "🟡"),
            },
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug, Serialize, Deserialize)]
pub enum GameStatus {
    InProgress,
    Finished(TokenColor),
}

#[derive(Debug, Clone)]
pub struct MoveError {
    message: String,
}

impl fmt::Display for MoveError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "{}", self.message)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Board {
    pub rows: usize,
    pub columns: usize,
    fields: Vec<Vec<Field>>,
    pub columns_full: Vec<bool>,
}

impl Board {
    pub fn show(self: &Self) {
        self.fields.iter().for_each(|row: &Vec<Field>| {
            row.iter().for_each(|field: &Field| print!("{} ", field));
            println!()
        });
        println!()
    }

    pub fn make_move(self: &mut Self, column: usize, color: TokenColor) -> Result<(), MoveError> {
        if !self.is_move_legal(column) {
            return Err(MoveError {
                message: "illegal move".to_string(),
            });
        }

        for row in (0..self.rows).rev() {
            match self.fields[row][column] {
                Field::Empty => {
                    self.fields[row][column] = Field::Token(color);
                    if row == 0 {
                        self.columns_full[column] = true;
                    }

                    break;
                }

                Field::Token(_) => continue,
            }
        }

        return Ok(());
    }

    pub fn undo_move(self: &mut Self, column: usize) -> Result<(), MoveError> {
        if column >= self.columns {
            return Err(MoveError {
                message: "column index out of range".to_string(),
            });
        }

        if self.fields[self.rows - 1][column] == Field::Empty {
            return Err(MoveError {
                message: "column is empty".to_string(),
            });
        }

        for i in 0..self.rows {
            if self.fields[i][column] != Field::Empty {
                self.fields[i][column] = Field::Empty;
                break;
            }
        }

        if self.fields[0][column] == Field::Empty {
            self.columns_full[column] = false;
        }

        return Ok(());
    }

    pub fn get_status(self: &Self) -> GameStatus {
        return match self.get_winner() {
            Some(color) => GameStatus::Finished(color),
            None => GameStatus::InProgress,
        };
    }

    pub fn is_move_legal(self: &Self, column: usize) -> bool {
        if column >= self.columns {
            return false;
        }

        if self.columns_full[column] {
            return false;
        }

        return true;
    }

    fn get_winner(self: &Self) -> Option<TokenColor> {
        for row in 0..self.rows {
            for column in 0..self.columns {
                if self.fields[row][column] == Field::Empty {
                    continue;
                }

                // Rows
                if (column <= self.columns - 4)
                    && (self.fields[row][column] == self.fields[row][column + 1])
                    && (self.fields[row][column + 1] == self.fields[row][column + 2])
                    && (self.fields[row][column + 2] == self.fields[row][column + 3])
                {
                    return match self.fields[row][column] {
                        Field::Token(color) => Some(color),
                        Field::Empty => None,
                    };
                }

                // Columns
                if (row <= self.rows - 4)
                    && (self.fields[row][column] == self.fields[row + 1][column])
                    && (self.fields[row + 1][column] == self.fields[row + 2][column])
                    && (self.fields[row + 2][column] == self.fields[row + 3][column])
                {
                    return match self.fields[row][column] {
                        Field::Token(color) => Some(color),
                        Field::Empty => None,
                    };
                }

                // Left diagonal
                if (row >= 3 && column >= 3)
                    && (self.fields[row][column] == self.fields[row - 1][column - 1])
                    && (self.fields[row - 1][column - 1] == self.fields[row - 2][column - 2])
                    && (self.fields[row - 2][column - 2] == self.fields[row - 3][column - 3])
                {
                    return match self.fields[row][column] {
                        Field::Token(color) => Some(color),
                        Field::Empty => None,
                    };
                }

                // Right diagonal
                if (row >= 3 && column <= self.columns - 4)
                    && (self.fields[row][column] == self.fields[row - 1][column + 1])
                    && (self.fields[row - 1][column + 1] == self.fields[row - 2][column + 2])
                    && (self.fields[row - 2][column + 2] == self.fields[row - 3][column + 3])
                {
                    return match self.fields[row][column] {
                        Field::Token(color) => Some(color),
                        Field::Empty => None,
                    };
                }
            }
        }

        return None;
    }
}

impl Default for Board {
    fn default() -> Self {
        return Self {
            rows: DEFAULT_ROWS,
            columns: DEFAULT_COLUMNS,
            fields: vec![vec![Field::Empty; DEFAULT_COLUMNS]; DEFAULT_ROWS],
            columns_full: vec![false; DEFAULT_COLUMNS],
        };
    }
}
