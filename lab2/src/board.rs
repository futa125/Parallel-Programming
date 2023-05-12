use std::{fmt, result, cmp};

const DEFAULT_ROWS: usize = 6;
const DEFAULT_COLUMNS: usize = 7;
const CONNECT_COUNT: usize = 4;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TokenColor {
    Red,
    Yellow,
}

#[derive(Clone, PartialEq, Eq)]
enum Field {
    Empty,
    Marker(TokenColor),
}

impl fmt::Display for Field {
    fn fmt(&self, f: &mut fmt::Formatter) -> result::Result<(), fmt::Error> {
        match self {
            Field::Empty => f.write_str("⚪"),
            Field::Marker(color) => match color {
                TokenColor::Red => write!(f, "🔴"),
                TokenColor::Yellow => write!(f, "🟡"),
            },
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum GameStatus {
    InProgress,
    Finished(TokenColor),
}

#[derive(Debug, Clone)]
pub struct MoveError {
    message: String,
}

impl fmt::Display for MoveError {
    fn fmt(&self, f: &mut fmt::Formatter) -> ::std::result::Result<(), ::std::fmt::Error> {
        write!(f, "{}", self.message)
    }
}

pub struct Board {
    rows: usize,
    columns: usize,
    fields: Vec<Vec<Field>>,
    columns_full: Vec<bool>,
    status: GameStatus,
}

impl Board {
    pub fn new(rows: usize, columns: usize) -> Self {
        return Self {
            rows,
            columns,
            fields: vec![vec![Field::Empty; columns]; rows],
            columns_full: vec![false; columns],
            status: GameStatus::InProgress,
        };
    }

    pub fn show(self: &Self) {
        self.fields.iter().for_each(|row: &Vec<Field>| {
            row.iter().for_each(|field: &Field| print!("{} ", field));
            println!()
        });
        println!()
    }

    pub fn make_move(
        self: &mut Self,
        column: usize,
        color: TokenColor,
    ) -> Result<GameStatus, MoveError> {
        if !self.is_move_legal(column) {
            return Err(MoveError {
                message: "illegal move".to_string(),
            });
        }

        match self.status {
            GameStatus::Finished(_) => {
                return Err(MoveError {
                    message: "game finished, can't make additional moves".to_string(),
                })
            }
            _ => (),
        }

        for row in (0..self.rows).rev() {
            match self.fields[row][column] {
                Field::Empty => {
                    self.fields[row][column] = Field::Marker(color);
                    if row == 0 {
                        self.columns_full[column] = true;
                    }

                    break;
                }

                Field::Marker(_) => continue,
            }
        }

        if self.is_move_winning(color) {
            let status = GameStatus::Finished(color);
            self.status = status;

            return Ok(status);
        }

        return Ok(GameStatus::InProgress);
    }

    fn is_move_legal(self: &Self, column: usize) -> bool {
        if column >= self.columns {
            return false;
        }

        if self.columns_full[column] {
            return false;
        }

        return true;
    }

    fn is_move_winning(self: &Self, color: TokenColor) -> bool {
        if self.is_move_winning_rows(color) {
            return true;
        }

        if self.is_move_winning_columns(color) {
            return true;
        }

        if self.is_move_winning_diagonals(color) {
            return true;
        }

        return false;
    }

    fn is_move_winning_rows(self: &Self, color: TokenColor) -> bool {
        let mut counter: usize = 0;

        for row in 0..self.rows {
            for column in 0..self.columns {
                if self.markers_connected(row, column, color, &mut counter) {
                    return true;
                };
            }

            counter = 0;
        }

        return false;
    }

    fn is_move_winning_columns(self: &Self, color: TokenColor) -> bool {
        let mut counter: usize = 0;

        for column in 0..self.columns {
            for row in 0..self.rows {
                if self.markers_connected(row, column, color, &mut counter) {
                    return true;
                };
            }

            counter = 0;
        }

        return false;
    }

    fn is_move_winning_diagonals(self: &Self, color: TokenColor) -> bool {
        let mut start_column: usize;
        let mut diagonal_size: usize;
        let mut counter_left: usize = 0;
        let mut counter_right: usize = 0;

        let mut row: usize;
        let mut column_left: usize;
        let mut column_right: usize;

        for diagonal in 1..self.rows + self.columns {
            if self.rows > diagonal {
                start_column = 0;
            } else {
                start_column = cmp::max(0, diagonal - self.rows);
            }

            diagonal_size = cmp::min(diagonal, self.columns - start_column);
            diagonal_size = cmp::min(diagonal_size, self.rows);

            for i in 0..diagonal_size {
                row = cmp::min(self.rows, diagonal) - i - 1;
                column_left = self.columns - (start_column + i) - 1;
                column_right = start_column + i;

                if self.markers_connected(row, column_left, color, &mut counter_left) {
                    return true;
                };

                if self.markers_connected(row, column_right, color, &mut counter_right) {
                    return true;
                };
            }

            counter_left = 0;
            counter_right = 0;
        }

        return false;
    }

    fn markers_connected(
        self: &Self,
        row: usize,
        column: usize,
        color: TokenColor,
        counter: &mut usize,
    ) -> bool {
        if self.fields[row][column] == Field::Marker(color) {
            *counter += 1;
            if *counter == CONNECT_COUNT {
                return true;
            }
        } else {
            *counter = 0;
        }

        return false;
    }
}

impl Default for Board {
    fn default() -> Self {
        return Self {
            rows: DEFAULT_ROWS,
            columns: DEFAULT_COLUMNS,
            fields: vec![vec![Field::Empty; DEFAULT_COLUMNS]; DEFAULT_ROWS],
            columns_full: vec![false; DEFAULT_COLUMNS],
            status: GameStatus::InProgress,
        };
    }
}
