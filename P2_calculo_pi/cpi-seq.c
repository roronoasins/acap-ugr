#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

main(int argc, char **argv)
{
  register double width, sum;
  register int intervals, i;

  /* get the number of intervals */
  intervals = atoi(argv[1]);
  width = 1.0 / intervals;

  double start_time = omp_get_wtime();
  /* do the computation */
  sum = 0;
  for (i=0; i<intervals; ++i) {
    register double x = (i + 0.5) * width;
    sum += 4.0 / (1.0 + x * x);
  }
  sum *= width;

  double time = omp_get_wtime() - start_time;

  printf("Estimation of pi is %f\nTiempo_secuencial:\t%8.10f\nIntervalos:\t%d\n", sum, time,intervals);

  return(0);
}
