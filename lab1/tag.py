from enum import IntEnum, auto


class Tag(IntEnum):
    REQUEST_LEFT_FORK = auto()
    REQUEST_RIGHT_FORK = auto()
    SEND_LEFT_FORK = auto()
    SEND_RIGHT_FORK = auto()
