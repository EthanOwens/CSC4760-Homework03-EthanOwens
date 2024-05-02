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

    int ele_per_proc = M / Q;

    Kokkos::View<int**> A("A", M, N);
    Kokkos::parallel_for("init_A", M, KOKKOS_LAMBDA(int i) {
        Kokkos::parallel_for("init_A", N, KOKKOS_LAMBDA(int j) {
            A(i, j) = i * j; 
        });
    });
    Kokkos::fence();

    Kokkos::View<int*> x("x", N);
    Kokkos::parallel_for("fill x ", N, KOKKOS_LAMBDA(int i){
        x(i) = i;
    });
    Kokkos::fence();

    MPI_Bcast(x.data(), N, MPI_INT, 0, MPI_COMM_WORLD);

    Kokkos::View<int*> local_y("y Local", ele_per_proc);
    Kokkos::parallel_for("compute y i", ele_per_proc, KOKKOS_LAMBDA(int i) {
        Kokkos::parallel_for("compute y j", ele_per_proc, KOKKOS_LAMBDA(int j) {
            sum += A(i + row * ele_per_proc, j + col * ele_per_proc) * x(j + col * ele_per_proc);
        });
        local_y(i) = sum;
    });
    Kokkos::fence();

    Kokkos::View<int*> y("y", M);
    MPI_Gather(local_y.data(), ele_per_proc, MPI_INT, y.data(), ele_per_proc, MPI_INT, 0, MPI_COMM_WORLD);
    
    if(rank == 0){
        printf("Ax product: \n");
        for(int i = 0; i < M; i++){
            printf("y = %d", y(i));
        }
    }

    
    MPI_Finalize();
    Kokkos::finalize();
}