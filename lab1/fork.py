from dataclasses import dataclass
from enum import Enum, auto


class ForkState(Enum):
    CLEAN = auto()
    DIRTY = auto()


@dataclass
class Fork:
    state: ForkState = ForkState.DIRTY
