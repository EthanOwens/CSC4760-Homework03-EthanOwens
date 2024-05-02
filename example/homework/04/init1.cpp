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

    Kokkos::View<int*> x_horiz("horizontal x", M);

    if(rank == 0){
        Kokkos::parallel_for("fill x horiz", M, KOKKOS_LAMBDA(int i){
            x_horiz(i) = i;
        });
    }
    Kokkos::fence();

    Kokkos::View<int*> local_x_horiz("horizontal x Local", ele_per_proc);
    Kokkos::parallel_for("scatter x horizontally", ele_per_proc, KOKKOS_LAMBDA(int i) {
        local_x_horiz(i) = x_horiz(i * Q + my_col);
    });
    Kokkos::fence();

    
    // MPI_Scatter(local_x.data(), M / P, MPI_INT, local_x.data(), M / P, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(local_x_horiz.data(), ele_per_proc, MPI_INT, col, MPI_COMM_WORLD);

    Kokkos::View<int*> x_vert("vertical x", M);
    if(rank == 0){
        Kokkos::parallel_for("fill x vert", M, KOKKOS_LAMBDA(int i){
            x_vert(i) = i;
        });
    }
    Kokkos::fence();

    Kokkos::View<int*> local_x_vert = Kokkos::subview(x_vert, Kokkos::ALL, Kokkos::subview::range(0, ele_per_proc));
    Kokkos::parallel_for("scatter x vertical", ele_per_proc, KOKKOS_LAMBDA(int i) {
        local_x_vert(i) = x_vert(i * P + row);
    });
    Kokkos::fence();

    MPI_Bcast(local_x_vert.data(), ele_per_proc, MPI_INT, row, MPI_COMM_WORLD);

    Kokkos::View<int> dot_product("dot product");
    Kokkos::parallel_reduce("dot_product", ele_per_proc, KOKKOS_LAMBDA(int i, int& local_sum) {
        local_sum += local_x_horiz(i) * local_x_vert(i);
    }, Kokkos::Sum<int>(dot_product));
    Kokkos::fence();

    int global_dot_product;
    MPI_Allreduce(&dot_product, &global_dot_product, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    
    MPI_Finalize();
    Kokkos::finalize();
}