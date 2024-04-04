#include <iostream>

#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    if (0 == rank) {
        int *buf = new int[numRanks];

        // emulate MPI_Gather to show send and receive
        for (int i = 1; i < numRanks; ++i)
            MPI_Recv(&(buf[i]), 1, MPI_INT, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // emulate MPI_Reduce
        buf[0] = rank;
        for (int i = 1; i < numRanks; ++i)
            buf[0] += buf[i];

        std::cout << "The sum of all ranks is " << buf[0] << std::endl;

        delete[] buf;
    } else {
        MPI_Send(&rank, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
}
