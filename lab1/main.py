import logging

from mpi4py import MPI

from philosopher import Philosopher, Side


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='[%(process)d][%(levelname)s][%(asctime)s] %(message)s',
        datefmt='%d-%b-%y %H:%M:%S',
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
                philosopher.process_incoming_responses()
                philosopher.process_incoming_requests()

            philosopher.send_fork_request(Side.RIGHT)

            while philosopher.right_fork is None:
                philosopher.process_incoming_responses()
                philosopher.process_incoming_requests()

        philosopher.eat()
        philosopher.process_backlog_requests()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
