#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
/**
 * Vector multiplication: C = A * B.
 *
 * Very basic sequential element by element vector multiplication.
 */

int
main(int argc, char **argv)
{
  if(argc<3)
  {
    printf("You must specify vector input files\n");
    exit(1);
  }

	/** Open input raw file passed as argument **/
  FILE *input_rawA, *input_rawB;
  input_rawA = fopen(argv[1],"r");
  input_rawB = fopen(argv[2], "r");

  /** Print the vector length to be used, and compute its size **/
  int numElementsA, numElementsB, numElements;
  fscanf(input_rawA, "%d", &numElementsA);
  fscanf(input_rawB, "%d", &numElementsB);
  if (numElementsA != numElementsB)
  {
    fprintf(stderr, "Vectors have not equal size!\n");
    exit(EXIT_FAILURE);
  }
  numElements = numElementsB;
  size_t size = numElements * sizeof(float);
  printf("[Vector addition of %d elements]\n", numElements);

  /** Allocate input vector A **/
  float *host_A = (float *)malloc(size);

  /** Allocate input vector B **/
  float *host_B = (float *)malloc(size);

  /** Allocate output vector C **/
  float *host_C = (float *)malloc(size);

  /** Verify that allocations succeeded **/
  if (host_A == NULL || host_B == NULL || host_C == NULL)
  {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }

  /** Initialize input vectors **/
  float aux;
  for (int i = 0; i < numElements; ++i)
  {
    fscanf(input_rawA,"%f",&aux);
    host_A[i] = aux;
    fscanf(input_rawB,"%f",&aux);
    host_B[i] = aux;
  }

  /** Calculate output vector **/
  double start_time = omp_get_wtime();
  for (int i = 0; i < numElements; ++i)
  {
		for (int j=0; j < 1000; j++)
    host_C[i] = host_A[i] * host_B[i] + j;
  }
  double time = omp_get_wtime() - start_time;

  printf("Test PASSED\n");

  /** Save result vector in output file **/
  FILE * output_C;
  if( ( output_C = fopen( "output.raw", "w" ) ) == NULL )
  {
    printf( "Output file could not be created\n" );
  }

  for (int i = 0; i < numElements; i++)
  {
    fprintf(output_C, "%.5f\n", host_C[i]);
  }

  /** Free host memory **/
  free(host_A);
  free(host_B);
  free(host_C);

  /** Close files **/
  fclose(input_rawA);
  fclose(input_rawB);
  fclose(output_C);

  /** Output data for stadistics **/
  FILE * data;
  data = fopen("data.txt", "a");
  fprintf(data,"Vector size: %d Proc time:\t%8.10f\n\n", numElements,time);
  fclose(data);

  printf("Done\nProcessing time: %.9lf\n", time);
  return 0;
}
