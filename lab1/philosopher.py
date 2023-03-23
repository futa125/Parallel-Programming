import logging
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from mpi4py import MPI

from fork import Fork, ForkState
from tag import Tag

MIN_SLEEP_DURATION = 1
MAX_SLEEP_DURATION = 5


@dataclass
class Requests:
    request_left: bool = False
    request_right: bool = False


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

        logging.info(f"{self._get_indent()}Process started")

    def think(self) -> None:
        sleep_duration = random.randint(MIN_SLEEP_DURATION, MAX_SLEEP_DURATION)

        logging.info(f"{self._get_indent()}Thinking for {sleep_duration} seconds")
        for _ in range(sleep_duration):
            time.sleep(1)
            self.process_incoming_requests()

    def process_incoming_requests(self) -> None:
        self._process_incoming_request_left()
        self._process_incoming_request_right()

    def send_fork_requests(self) -> None:
        self._send_fork_request_left()
        self._send_fork_request_right()

    def process_incoming_responses(self):
        self._process_incoming_response_left()
        self._process_incoming_response_right()

    def eat(self):
        logging.info(f"{self._get_indent()}Eating")
        self.left_fork.state = ForkState.DIRTY
        self.right_fork.state = ForkState.DIRTY

    def process_backlog_requests(self):
        if self.backlog_requests.request_left:
            self._send_right_fork()
            self.backlog_requests.request_left = False

        if self.backlog_requests.request_right:
            self._send_left_fork()
            self.backlog_requests.request_right = False

    def _process_incoming_request_left(self) -> None:
        if self.comm.Iprobe(source=self.right_neighbor_rank, tag=Tag.REQUEST_LEFT):
            self.comm.recv(source=self.right_neighbor_rank, tag=Tag.REQUEST_LEFT)
            logging.info(f"{self._get_indent()}Received a request for my right fork")

            if self.right_fork is not None and self.right_fork.state == ForkState.DIRTY:
                self._send_right_fork()
            else:
                self.backlog_requests.request_left = True

    def _process_incoming_request_right(self):
        if self.comm.Iprobe(source=self.left_neighbor_rank, tag=Tag.REQUEST_RIGHT):
            self.comm.recv(source=self.left_neighbor_rank, tag=Tag.REQUEST_RIGHT)
            logging.info(f"{self._get_indent()}Received a request for my left fork")

            if self.left_fork is not None and self.left_fork.state == ForkState.DIRTY:
                self._send_left_fork()
            else:
                self.backlog_requests.request_right = True

    def _send_left_fork(self):
        self.left_fork.state = ForkState.CLEAN
        self.comm.isend(self.left_fork, dest=self.left_neighbor_rank, tag=Tag.SEND_LEFT)
        logging.info(f"{self._get_indent()}Sent my left fork")

        self.left_fork = None

    def _send_right_fork(self):
        self.right_fork.state = ForkState.CLEAN
        self.comm.isend(self.right_fork, dest=self.right_neighbor_rank, tag=Tag.SEND_RIGHT)
        logging.info(f"{self._get_indent()}Sent my right fork")

        self.right_fork = None

    def _send_fork_request_left(self):
        if self.left_fork is not None:
            return

        if self.sent_requests.request_left:
            return

        self.comm.isend(None, dest=self.left_neighbor_rank, tag=Tag.REQUEST_LEFT)
        logging.info(f"{self._get_indent()}Sent a request for my left fork")
        self.sent_requests.request_left = True

    def _send_fork_request_right(self):
        if self.right_fork is not None:
            return

        if self.sent_requests.request_right:
            return

        self.comm.isend(None, dest=self.right_neighbor_rank, tag=Tag.REQUEST_RIGHT)
        logging.info(f"{self._get_indent()}Sent a request for my right fork")
        self.sent_requests.request_right = True

    def _process_incoming_response_left(self):
        if self.comm.Iprobe(source=self.right_neighbor_rank, tag=Tag.SEND_LEFT):
            self.right_fork = self.comm.recv(source=self.right_neighbor_rank, tag=Tag.SEND_LEFT)
            logging.info(f"{self._get_indent()}Received my right fork")
            self.sent_requests.request_right = False

    def _process_incoming_response_right(self):
        if self.comm.Iprobe(source=self.left_neighbor_rank, tag=Tag.SEND_RIGHT):
            self.left_fork = self.comm.recv(source=self.left_neighbor_rank, tag=Tag.SEND_RIGHT)
            logging.info(f"{self._get_indent()}Received my left fork")
            self.sent_requests.request_left = False

    def _get_indent(self) -> str:
        return "    " * self.rank
