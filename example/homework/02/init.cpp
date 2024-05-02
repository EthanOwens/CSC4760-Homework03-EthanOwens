//NOTE: Pieces of code that were added are tagged with "//ADDED CODE BLOCK"

using namespace std;
#include <iostream>
#include <assert.h>
#include <string>

#include <mpi.h>
#include "Distributions.h"  // contains LinearDistribution class.
#include "Process_2DGrid.h" // defines Process_2DGrid class.

#define GLOBAL_PRINT  // gather and print domains from process rank 0

// forward declarations:

class Domain // this will be integrated with a Kokkos view and subview in HW#3.
{            // this is a row-major arrangement of data in 2D.
public:
  Domain(int _M, int _N, int _halo_depth, const char *_name="") :
        exterior(new char[(_M+2*_halo_depth)*(_N+2*_halo_depth)]), M(_M), N(_N),
	halo_depth(_halo_depth), name(_name)
        {interior = exterior+(N+2*halo_depth)*halo_depth + halo_depth;}
  
  virtual ~Domain() {delete[] exterior;}
  
  char &operator()(int i, int j)       {return interior[i*(N+2*halo_depth)+j];}
  char operator() (int i, int j) const {return interior[i*(N+2*halo_depth)+j];}

  int rows() const {return M;}
  int cols() const {return N;} // actual domain size.

  const string &myname() const {return name;}

  char *rawptr()    {return exterior;}
  char *cookedptr() {return interior;}  

protected:
  char *exterior, *interior;
  int M;
  int N;
  int halo_depth;

  string name;
};



// more forward declarations:
void zero_domain  (Domain &domain); // zeros the halos too.
void print_domain (Domain &domain, int p, int q); // prints only the true domain
void update_domain(Domain &new_domain, Domain &old_domain, Process_2DGrid &grid);
void parallel_code(int M, int N, int iterations, Process_2DGrid &grid);

int main(int argc, char **argv)
{
  // command-line-specified parameters:
  int M, N;
  int P, Q;
  int iterations;

  if(argc < 6)
  {
    cout << "usage: " << argv[0] << " M N P Q iterations" << endl;
    exit(0);
  }

  int size, myrank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  int array[5];
  if(0 == myrank)
  {
     M = atoi(argv[1]); N = atoi(argv[2]);
     P = atoi(argv[3]); Q = atoi(argv[4]);
     iterations = atoi(argv[5]);

     array[0] = M;
     array[1] = N;
     array[2] = P;
     array[3] = Q;
     array[4] = iterations;
  }
  MPI_Bcast(array, 5, MPI_INT, 0, MPI_COMM_WORLD);
  if(myrank != 0)
  {
    M = array[0];
    N = array[1];
    P = array[2];
    Q = array[3];
    iterations = array[4];
  }

  if(P*Q != size)
  {
    if(0==myrank)
      cout << "grid mismatch: " << argv[0] << P << "*" << Q << "must be size of world: "<< size <<"!" << endl;
    
    exit(0);
  }
  
  Process_2DGrid grid(P, Q, MPI_COMM_WORLD);
  parallel_code(M, N, iterations, grid);
  
  MPI_Finalize();
}

void parallel_code(int M, int N, int iterations, Process_2DGrid &grid)
{
  // use the linear, load-balanced distribution in both dimensions:
  LinearDistribution row_distribution(M, grid.P());
  LinearDistribution col_distribution(N, grid.Q());

  // mxn is the size of the domain (p,q):
  int m = row_distribution.m(grid.p());
  int n = col_distribution.m(grid.q());

  // local domains (replace with Kokkos View and Subview in Kokkos version!):
  Domain even_domain(m,n,1,"even local domain"); // halo depths of 1
  Domain odd_domain (m,n,1,"odd local domain");  // mxn size, varies per p,q location

  zero_domain(even_domain);
  zero_domain(odd_domain);

#ifdef GLOBAL_PRINT // not scalable in memory or I/O to print each iteration, for testing.
  Domain *global_domain = nullptr;
  Domain *domain_slice  = nullptr;
  int    *rowcounts     = nullptr;
  int    *rowdispls     = nullptr;
  int    *colcounts     = nullptr;
  int    *coldispls     = nullptr;

  if(grid.q() == 0) // left column of grid only does this distribution. p=0...,P-1
  {
    domain_slice = new Domain(m,N,0,"slice of domain"); // no halos, 1D decomp.
                                                        // mxN size
    
    if(grid.p() == 0) // root of first [vertical] scatter
    {
      rowcounts = new int[grid.P()];
      rowdispls = new int[grid.P()];

      for(int phat = 0; phat < grid.P(); ++phat)
	rowcounts[phat] = row_distribution.m(phat)*col_distribution.M();  //whole cols.

      int count = rowdispls[0] = 0;
      for(int phat = 1; phat < grid.P(); ++phat)
      {
        count += rowcounts[phat-1];
        rowdispls[phat] = count;
      }

      // process (0,0) holds an initial copy of the whole domain [to illustrate
      // scatter, to help with I/O.  you don't do this in really scalable programs.]
      global_domain = new Domain(M,N,0,"Global Domain"); // no halos, MxN size
      zero_domain(*global_domain);

      if((N >= 8) && (M >= 10))
      {
	// blinker at top left, touching right...
	(*global_domain)(0,(N-1)) = 1;
	(*global_domain)(0,0)     = 1;
	(*global_domain)(0,1)     = 1;

	// and a glider:
	(*global_domain)(8,5)     = 1;
	(*global_domain)(8,6)     = 1;
	(*global_domain)(8,7)     = 1;
	(*global_domain)(7,7)     = 1;
	(*global_domain)(6,6)     = 1;
      }

    } // if(grid.p() == 0

    MPI_Scatterv((grid.p()==0) ? global_domain->rawptr() : nullptr,
		 rowcounts, rowdispls, MPI_CHAR, domain_slice->rawptr(),
                 m*N, MPI_CHAR, 0 /* root */, grid.col_comm() /*vertical split */);
    // we are re-using rowcounts, rowdispls, global_domain, domain_slide
    // in the print functions below.  Otherwise, we'd free them here.

    // process column zero has vertically scattered domain in 1D so far.
    // domain slice in leftmost column must be distributed into even local domain.
    // now work to scatter domain_slice horizontally in each process row.
    
    colcounts = new int[grid.Q()];
    coldispls = new int[grid.Q()];

    for(int qhat = 0; qhat < grid.Q(); ++qhat)
      colcounts[qhat] = col_distribution.m(qhat);  /* notice it is just col count*/

    int count = coldispls[0] = 0;
    for(int qhat = 1; qhat < grid.Q(); ++qhat)
    {
      count += m*colcounts[qhat-1]; // but this accounts for total values per process
      coldispls[qhat] = count;
    }
  } // end if(grid.q() == 0)
 
  // processes with grid.q()==0 use this:
  MPI_Datatype vertical_slice_in = MPI_DATATYPE_NULL;  // notice global stride of N
  if(grid.q()==0)
  {
     MPI_Type_vector(m /* count */, 1 /* blocklen */, N /* stride, no halos */,
    		     MPI_CHAR, &vertical_slice_in);

     MPI_Type_commit(&vertical_slice_in);
  }

  // all processes need this:
  MPI_Datatype vertical_slice_out = MPI_DATATYPE_NULL; // notice local stride of n+2
  MPI_Type_vector(m /* count */, 1 /* blocklen */, n+2*1 /* stride incl halos(1) */,
		  MPI_CHAR, &vertical_slice_out);
  MPI_Type_commit(&vertical_slice_out);

  // we point at the interior of the even_domain as the base for transfer:
  MPI_Scatterv((grid.q()==0) ? domain_slice->rawptr() : nullptr,
	       colcounts, coldispls, vertical_slice_in, even_domain.cookedptr(), n,
	       vertical_slice_out, 0 /* root */, grid.row_comm());

#else
  // locally initialize domains if not GLOBAL_PRINT.
#endif
  
#ifdef GLOBAL_PRINT
    if(grid.p()==0 && grid.q()==0)
    {  
      cout << "Initial State:" << endl;
      print_domain(*global_domain, 0,0);
    }
#else
    cout << "Initial State:" << i << endl;
    print_domain(*even, p, q);
#endif  

  Domain *odd, *even; // pointer swap magic
  odd  = &odd_domain;
  even = &even_domain;

  for(int i = 0; i < iterations; ++i)
  {
    update_domain(*odd, *even, grid);

#ifdef GLOBAL_PRINT

    // gather the domains from all PxQ processes back into global_domain,
    // but not their halos... the data to print is in the odd ptr.

#if 0  // reverse the senses from Scatterv's to Gatherv's.. [datatypes exist]
    //MPI_Scatterv((grid.q()==0) ? domain_slice->rawptr() : nullptr,
    //		 colcounts, coldispls, vertical_slice_in, even_domain.cookedptr(), n,
    //   	 vertical_slice_out, 0 /* root */, grid.row_comm());

    //API:
    //int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
    //            void *recvbuf, const int recvcounts[], const int displs[],
    //            MPI_Datatype recvtype, int root, MPI_Comm comm)
    MPI_Gatherv(odd->cookedptr(), n, vertical_slice_out, 
		(grid.q()==0) ? domain_slice->rawptr() : nullptr, 
		vertical_slice_in, colcounts, coldispls, vertical_slice_in, 
		0 /* root */, grid.row_comm());

    if(grid.q() == 0)
    {
      // reverse:
      //MPI_Scatterv((grid.p()==0) ? global_domain->rawptr() : nullptr,
      //           rowcounts, rowdispls, MPI_CHAR, domain_slice->rawptr(),
      //	   m*N, MPI_CHAR, 0 /* root */, grid.col_comm() /*vertical split */);
      
      //API:
      //int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
      //            void *recvbuf, const int recvcounts[], const int displs[],
      //            MPI_Datatype recvtype, int root, MPI_Comm comm)
      MPI_Gatherv(domain_slice->rawptr(), m*N, MPI_CHAR,
		  (grid.p()==0) ? global_domain->rawptr() : nullptr,
		  rowcounts, rowdispls, 0 /* root */, grid.col_comm());

    } // end if(grid.q() == 0)

#endif 
    if(grid.p()==0 && grid.q()==0)
    {
      cout << "Iteration #" << i << endl; 
      print_domain(*global_domain, 0, 0);
    }
#else
    cout << "Iteration #" << i << endl; print_domain(*odd, p, q);
#endif

    // swap pointers:
    Domain *temp = odd;
    odd  = even;
    even = temp;
  }
  
#ifdef GLOBAL_PRINT
  if(grid.p() == 0 && grid.q() == 0)
  {
    delete global_domain;
  }

  if(grid.q() == 0) // free memory & datatypes associated with first scatter/gather
  {
    MPI_Type_free(&vertical_slice_in);

    delete[] domain_slice;
    delete[] colcounts;
    delete[] coldispls;
  }

  if(grid.p() == 0) // free memory & datatypes associated with the second scatter/gather
  {
    delete[] rowcounts;
    delete[] rowdispls;
  }

  // free any other memory & datatypes applying to all (p,q)...
  MPI_Type_free(&vertical_slice_out);
  
#endif  
}

void zero_domain(Domain &domain)
{
  for(int i = 0; i < domain.rows(); ++i)
    for(int j = 0; j < domain.cols(); ++j)
      domain(i,j) = 0;
}

void print_domain(Domain &domain, int p, int q)
{
  cout << "(" << p << "," << q << "): " << domain.myname() << ":" <<endl;
  // this is naive; it doesn't understand big domains at all 
  for(int i = 0; i < domain.rows(); ++i)
  {
    for(int j = 0; j < domain.cols(); ++j)
      cout << (domain(i,j) ? "*" : ".");
    cout << endl;
  }
}

inline char update_the_cell(char cell, int neighbor_count) //Life cellular a rule.
{
  char newcell;
#if 1
  if(cell == 0) // dead now
    newcell = (neighbor_count == 3) ? 1 : 0;
  else // was live, what about now?
    newcell = ((neighbor_count == 2)||(neighbor_count == 3)) ? 1 : 0;
#else
  // alt logic: notice the cell is always live next generation, if it has three neighbors
  if(neighbor_count == 3)
    newcell = 1;
  else if((cell==1)&&(neighbor_count == 2)) newcell = 1;
  else
    newcell = 0;
#endif
  return newcell;
}

//ADDED CODE BLOCK
struct kernel_update {
	Domain new_domain;
	Domain old_domain;

	kernel_update(Domain new_dom, Domain old_dom) : new_domain(new_dom), old_domain(old_dom) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(int idx){
		int i = idx / new_domain.cols();
		int j = idx % new_domain.cols();

		int neighbor_count = 
			old_domain(row - 1, col - 1) + old_domain(row - 1, col) + old_domain(row - 1, col + 1) + 
			old_domain(row, col - 1) + 0 + old_domain(row, col + 1) +
            old_domain(row + 1, col - 1) + old_domain(row + 1, col) + old_domain(row + 1, col + 1);

		new_domain(row, col) = update_the_cell(old_domain(i, j), neighbor_count);
	}
};

//ENED ADDED CODE BLOCK
      
void update_domain(Domain &new_domain, Domain &old_domain, Process_2DGrid &grid)
{
  MPI_Request request[16];
  
  int m = new_domain.rows();
  int n = new_domain.cols();

  // this is used to avoid using MPI derived datatypes for now.
  char *left_col  = new char[m]; char *west_halo = new char[n];
  char *right_col = new char[m]; char *east_halo = new char[n]; 

  const int NORTH_HALO = 0, SOUTH_HALO = 1, WEST_HALO = 2, EAST_HALO = 3,
            NW_HALO    = 4, NE_HALO    = 5, SW_HALO   = 6, SE_HALO   = 7;

  const int North_prank = (grid.p()-1+grid.P())%grid.P();
  const int South_prank = (grid.p()+1         )%grid.P();
  const int West_qrank  = (grid.q()-1+grid.Q())%grid.Q();
  const int East_qrank  = (grid.q()+1         )%grid.Q();
  const int NW_rank     = grid.rank_of_pq((grid.p()-1+grid.P())%grid.P(),
				          (grid.q()-1+grid.Q())%grid.Q());
  const int NE_rank     = grid.rank_of_pq((grid.p()-1+grid.P())%grid.P(),
				          (grid.q()+1+grid.Q())%grid.Q());
  const int SW_rank     = grid.rank_of_pq((grid.p()+1+grid.P())%grid.P(),
				          (grid.q()-1+grid.Q())%grid.Q());
  const int SE_rank     = grid.rank_of_pq((grid.p()+1+grid.P())%grid.P(),
				          (grid.q()+1+grid.Q())%grid.Q());
  
  // use the col_comm and row_comm communicators for 4 cardinal direction:
  // these go directly into old_domain:
  MPI_Irecv(&old_domain(-1,0), n, MPI_CHAR, North_prank, NORTH_HALO, grid.col_comm(),
	    &request[0]);
  MPI_Irecv(&old_domain(m,0),  n, MPI_CHAR, South_prank, SOUTH_HALO, grid.col_comm(),
	    &request[1]);

  // these must be copied into strided part of old_domain:
  MPI_Irecv(west_halo, m, MPI_CHAR, West_qrank, WEST_HALO, grid.row_comm(), &request[2]);
  MPI_Irecv(east_halo, m, MPI_CHAR, East_qrank, EAST_HALO, grid.row_comm(), &request[3]);

  // corners:
  MPI_Irecv(&old_domain(-1,-1), 1, MPI_CHAR, NW_rank, NW_HALO, grid.parent_comm(),
	    &request[4]);
  MPI_Irecv(&old_domain(-1,n),  1, MPI_CHAR, NE_rank, NE_HALO, grid.parent_comm(),
	    &request[5]);
  MPI_Irecv(&old_domain(m,-1),  1, MPI_CHAR, SW_rank, SW_HALO, grid.parent_comm(),
	    &request[6]);
  MPI_Irecv(&old_domain(m,n),   1, MPI_CHAR, SE_rank, SE_HALO, grid.parent_comm(),
	    &request[7]);
  
  // contiguous, so send the data directly from data structure:
  MPI_Isend(&old_domain(0,0),  n, MPI_CHAR, North_prank, SOUTH_HALO, grid.col_comm(),
	    &request[8]);

  // contiguous, so send the data directly from data structure:
  MPI_Isend(&old_domain(m-1,0),n, MPI_CHAR, South_prank, NORTH_HALO, grid.col_comm(),
	    &request[9]);

  // these are strided, for now do copy:
  for(int i = 0; i < m; ++i)   // fill left 
     left_col[i] = old_domain(i,0);
  MPI_Isend(left_col, m, MPI_CHAR, West_qrank, EAST_HALO, grid.row_comm(), &request[10]);

  for(int i = 0; i < m; ++i)   // fill right
     right_col[i] = old_domain(i,n-1);
  MPI_Isend(right_col,m, MPI_CHAR, East_qrank, WEST_HALO, grid.row_comm(), &request[11]);

  MPI_Isend(&old_domain(m-1,n-1), 1, MPI_CHAR, SE_rank, NW_HALO, grid.parent_comm(),
	    &request[12]);//NW
  MPI_Isend(&old_domain(m-1,0),   1, MPI_CHAR, SW_rank, NE_HALO, grid.parent_comm(),
	    &request[13]);//NE
  MPI_Isend(&old_domain(0,n-1),   1, MPI_CHAR, NE_rank, SW_HALO, grid.parent_comm(),
	    &request[14]);//SW
  MPI_Isend(&old_domain(0,0),     1, MPI_CHAR, NW_rank, SE_HALO, grid.parent_comm(),
	    &request[15]);//SE

  // complete all 16 transfers
  MPI_Waitall(16, request, MPI_STATUSES_IGNORE);

  // the entire perimeter must now be copied into old_domain's halo space.
  for(int i = 0; i < m; ++i) 
  { 
    old_domain(i,-1) = west_halo[i];
    old_domain(i,n)  = east_halo[i];
  }
      
  // the perimeter containing all halos is now in old_domain;
  // we can compute on the interior of old_domain.

  // the entirety of the domain is computed so {replace with Kokkos kernel for HW#3}
  //ADDED COBE BLOCK
	int domain_size = new_domain.rows() * new_domain.cols();
	Kokkos::parallel_for(domain_size, kernel_update(new_domain, old_domain));
	Kokkos::fence();
  //ENED ADDED CODE BLOCK

  // remember, in a performant code, we would encapsulate the
  // dynamic memory allocation once level higher in the code...
  delete[] left_col,  west_halo;
  delete[] right_col, east_halo;
}


