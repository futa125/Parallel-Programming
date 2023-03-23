from enum import IntEnum, auto


class Tag(IntEnum):
    REQUEST_LEFT = auto()
    REQUEST_RIGHT = auto()
    SEND_LEFT = auto()
    SEND_RIGHT = auto()
