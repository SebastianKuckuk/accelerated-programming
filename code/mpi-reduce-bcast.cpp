#include <iostream>

#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc,&argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // emulate MPI_Allreduce to show different MPI operations:
    //  sum up all rank ids into rank 0
    //  then broadcast the sum to all ranks

    int sum = rank;
    MPI_Reduce(&rank, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    std::cout << "Sum on rank " << rank << " is " << sum << std::endl;

    // make sure all prints of stage one are done
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(&sum, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::cout << "Sum on rank " << rank << " is " << sum << std::endl;

    MPI_Finalize();
}
