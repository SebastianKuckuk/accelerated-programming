#include <iostream>

#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc,&argv);

    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    std::cout << "Hello from rank " << rank << " of " << numRanks << std::endl;

    MPI_Finalize();
}
