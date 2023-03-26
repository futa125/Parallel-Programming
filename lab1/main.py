import logging
import random
import time
from dataclasses import dataclass, field
from enum import auto, Enum, IntEnum
from typing import Optional

from mpi4py import MPI

MIN_SLEEP_DURATION = 1
MAX_SLEEP_DURATION = 5


class ForkState(Enum):
    CLEAN = auto()
    DIRTY = auto()


class Tag(IntEnum):
    REQUEST_LEFT_FORK = auto()
    REQUEST_RIGHT_FORK = auto()
    SEND_LEFT_FORK = auto()
    SEND_RIGHT_FORK = auto()


class Side(Enum):
    LEFT = auto()
    RIGHT = auto()


@dataclass
class Fork:
    state: ForkState = ForkState.DIRTY


@dataclass
class Requests:
    request_left_fork: bool = False
    request_right_fork: bool = False


@dataclass
class Philosopher:
    comm: MPI.Intracomm
    rank: int
    size: int

    left_fork: Optional[Fork] = field(init=False)
    right_fork: Optional[Fork] = field(init=False)

    left_neighbor_rank: int = field(init=False)
    right_neighbor_rank: int = field(init=False)

    backlog_requests: Requests = field(default_factory=Requests, init=False)
    sent_requests: Requests = field(default_factory=Requests, init=False)

    def __post_init__(self) -> None:
        if self.rank == 0:
            self.left_neighbor_rank, self.right_neighbor_rank = self.size - 1, self.rank + 1
        elif self.rank == self.size - 1:
            self.left_neighbor_rank, self.right_neighbor_rank = self.rank - 1, 0
        else:
            self.left_neighbor_rank, self.right_neighbor_rank = self.rank - 1, self.rank + 1

        if self.rank == 0:
            self.left_fork, self.right_fork = Fork(), Fork()
        elif self.rank == self.size - 1:
            self.left_fork, self.right_fork = None, None
        else:
            self.left_fork, self.right_fork = None, Fork()

        logging.debug(f"{self._get_indent()}Process started")

    def think(self) -> None:
        sleep_duration = random.randint(MIN_SLEEP_DURATION, MAX_SLEEP_DURATION)
        logging.debug(f"{self._get_indent()}Thinking for {sleep_duration} seconds")

        end_time = time.time() + sleep_duration

        while time.time() < end_time:
            self.process_fork_requests()

    def process_fork_requests(self) -> None:
        self._process_left_fork_request()
        self._process_right_fork_request()

    def send_fork_request(self, side: Side) -> None:
        if side == Side.LEFT:
            self._send_left_fork_request()
        elif side == Side.RIGHT:
            self._send_right_fork_request()

    def process_fork_responses(self) -> None:
        self._process_left_fork_response()
        self._process_right_fork_response()

    def eat(self) -> None:
        sleep_duration = random.randint(MIN_SLEEP_DURATION, MAX_SLEEP_DURATION)
        logging.debug(f"{self._get_indent()}Eating for {sleep_duration} seconds")
        time.sleep(sleep_duration)
        self.left_fork.state = ForkState.DIRTY
        self.right_fork.state = ForkState.DIRTY

    def process_backlog_fork_requests(self) -> None:
        if self.backlog_requests.request_left_fork:
            self._send_right_fork()
            self.backlog_requests.request_left_fork = False

        if self.backlog_requests.request_right_fork:
            self._send_left_fork()
            self.backlog_requests.request_right_fork = False

    def _process_left_fork_request(self) -> None:
        if self.comm.Iprobe(source=self.right_neighbor_rank, tag=Tag.REQUEST_LEFT_FORK):
            self.comm.recv(source=self.right_neighbor_rank, tag=Tag.REQUEST_LEFT_FORK)
            logging.debug(f"{self._get_indent()}Received a request for my right fork")

            if self.right_fork is not None and self.right_fork.state == ForkState.DIRTY:
                self._send_right_fork()
            else:
                self.backlog_requests.request_left_fork = True

    def _process_right_fork_request(self) -> None:
        if self.comm.Iprobe(source=self.left_neighbor_rank, tag=Tag.REQUEST_RIGHT_FORK):
            self.comm.recv(source=self.left_neighbor_rank, tag=Tag.REQUEST_RIGHT_FORK)
            logging.debug(f"{self._get_indent()}Received a request for my left fork")

            if self.left_fork is not None and self.left_fork.state == ForkState.DIRTY:
                self._send_left_fork()
            else:
                self.backlog_requests.request_right_fork = True

    def _send_left_fork(self) -> None:
        self.left_fork.state = ForkState.CLEAN
        self.comm.isend(self.left_fork, dest=self.left_neighbor_rank, tag=Tag.SEND_LEFT_FORK)
        logging.debug(f"{self._get_indent()}Sent my left fork")

        self.left_fork = None

    def _send_right_fork(self) -> None:
        self.right_fork.state = ForkState.CLEAN
        self.comm.isend(self.right_fork, dest=self.right_neighbor_rank, tag=Tag.SEND_RIGHT_FORK)
        logging.debug(f"{self._get_indent()}Sent my right fork")

        self.right_fork = None

    def _send_left_fork_request(self) -> None:
        if self.left_fork is not None:
            return

        if self.sent_requests.request_left_fork:
            return

        self.comm.isend(None, dest=self.left_neighbor_rank, tag=Tag.REQUEST_LEFT_FORK)
        logging.debug(f"{self._get_indent()}Sent a request for my left fork")
        self.sent_requests.request_left_fork = True

    def _send_right_fork_request(self) -> None:
        if self.right_fork is not None:
            return

        if self.sent_requests.request_right_fork:
            return

        self.comm.isend(None, dest=self.right_neighbor_rank, tag=Tag.REQUEST_RIGHT_FORK)
        logging.debug(f"{self._get_indent()}Sent a request for my right fork")
        self.sent_requests.request_right_fork = True

    def _process_left_fork_response(self) -> None:
        if self.comm.Iprobe(source=self.right_neighbor_rank, tag=Tag.SEND_LEFT_FORK):
            self.right_fork = self.comm.recv(source=self.right_neighbor_rank, tag=Tag.SEND_LEFT_FORK)
            logging.debug(f"{self._get_indent()}Received my right fork")
            self.sent_requests.request_right_fork = False

    def _process_right_fork_response(self) -> None:
        if self.comm.Iprobe(source=self.left_neighbor_rank, tag=Tag.SEND_RIGHT_FORK):
            self.left_fork = self.comm.recv(source=self.left_neighbor_rank, tag=Tag.SEND_RIGHT_FORK)
            logging.debug(f"{self._get_indent()}Received my left fork")
            self.sent_requests.request_left_fork = False

    def _get_indent(self) -> str:
        return "    " * self.rank


def main() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(process)d][%(levelname)s][%(asctime)s] %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    philosopher = Philosopher(comm, rank, size)

    while True:
        philosopher.think()

        while philosopher.left_fork is None or philosopher.right_fork is None:
            philosopher.send_fork_request(Side.LEFT)

            while philosopher.left_fork is None:
                philosopher.process_fork_responses()
                philosopher.process_fork_requests()

            philosopher.send_fork_request(Side.RIGHT)

            while philosopher.right_fork is None:
                philosopher.process_fork_responses()
                philosopher.process_fork_requests()

        philosopher.eat()
        philosopher.process_backlog_fork_requests()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
