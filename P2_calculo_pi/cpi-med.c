#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <omp.h>

#define PI 3.141592653589793238462643

main(int argc, char **argv)
{
  register double width, x, sum;
  register int intervals, i;
  double global_sum, local_sum;
  int np, total_np;
  MPI_Status status;

  intervals = atoi(argv[1]);
  double start_time = omp_get_wtime();
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &total_np);
  MPI_Comm_rank(MPI_COMM_WORLD, &np);

  width = 1.0 / intervals;
  local_sum = 0;

  for (i=np; i<intervals; i+=total_np) {
    x = (i + 0.5) * width;
    local_sum += 4.0 / (1.0 + x * x);
  }
  local_sum *= width;

  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Finalize();

  double parallel_time = omp_get_wtime() - start_time;
  double pi_error = (global_sum-PI)*100;

  if(!np)  printf(" Iterations:\t%d\n PI_value:\t%26.24f\n Procesos:\t%i\n Tiempo_paralelo:\t%8.10f\n Error:\t%8.20f%\n", intervals, global_sum, total_np, parallel_time, pi_error);

  return(0);
}
