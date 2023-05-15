use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum TokenColor {
    Red,
    Yellow,
}

impl TokenColor {
    pub fn invert(self: &Self) -> TokenColor {
        if *self == Self::Red {
            return Self::Yellow;
        }

        return Self::Red;
    }
}
