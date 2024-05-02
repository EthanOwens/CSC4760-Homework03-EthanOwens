#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <cstdio>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int P = 1, Q = 2, M = 3;

    //P*Q must equal world size
    if(size != P * Q){
        printf("Error: Number of processes must be equal to P * Q\n");
        MPI_Finalize();
        Kokkos::finalize();
        return 0;
    }

    int row = rankk / Q;
    int col = rank % Q;

    int global_J = rank;

    int j = global_J / Q;
    int q = global_J % Q;

    int ele_per_proc = M / Q;

    Kokkos::View<int*> vector_x("vx", M);

    if(rank == 0){
        Kokkos::parallel_for("fill x", M, KOKKOS_LAMBDA(int i){
            vector_x(i) = i;
        });
    }
    Kokkos::fence();

    Kokkos::View<int*> local_x("local_x", M / P);
    Kokkos::View<int*> temp_x = Kokkos::subview(vector_x, Kokkos::ALL, Kokkos::subview::range(0, M / P));
    Kokkos::deep_copy(local_x, temp_x);
    
    MPI_Scatter(local_x.data(), M / P, MPI_INT, local_x.data(), M / P, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(local_x.data(), M / P, MPI_INT, col, MPI_COMM_WORLD);

    Kokkos::View<int*> vector_y("vy", M);
    Kokkos::View<int*> sub_vector_y = Kokkos::subview(vector_y, Kokkos::ALL, Kokkos::subview::range(0, ele_per_proc));

    Kokkos::parallel_for("copy_y", ele_per_proc, KOKKOS_LAMBDA(int i) {
        sub_vector_y(i) = local_x(i);
    });

    Kokkos::fence();

    Kokkos::parallel_for("print y", M, KOKKOS_LAMBDA(int i) {
        if (i == 0) {
            printf("Process %d has y =", rank);
        }
        cout << vector_y(i);
        if (i == M - 1) {
            printf("\n");
        }
    });
    Kokkos::fence();

    
    MPI_Finalize();
    Kokkos::finalize();
}