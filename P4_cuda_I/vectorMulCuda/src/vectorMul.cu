/**
 * Implements element by element vector multiplication
 *
 * Vector multiplication: C = A * B.
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <omp.h>

/**
 * Computes the vector multiplication of A and B into C.
 * @param  A           [input vector]
 * @param  B           [input vector]
 * @param  C           [output vector]
 * @param  numElements [A, B, C number of elements]
 */
__global__ void vectorMul(const float *A, const float *B, float *C, int numElements)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < numElements)
  {
    for (int j=0; j < 1000; j++)
      C[i] = A[i] * B[i] + j;
  }
}

int main(int argc, char **argv)
{
  if(argc<3)
  {
    printf("You must specify vector input files\n");
    exit(1);
  }
  cudaError_t err = cudaSuccess;

  /** Open input raw file passed as argument **/
  FILE *input_rawA, *input_rawB;
  input_rawA = fopen(argv[1],"r");
  input_rawB = fopen(argv[2], "r");

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

  /** Allocate host vectors **/
  float *host_A = (float *)malloc(size);
  float *host_B = (float *)malloc(size);
  float *host_C = (float *)malloc(size);

  /** Verify that allocations succeeded **/
  if (host_A == NULL || host_B == NULL || host_C == NULL)
  {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }

  /** Initialize the host input vectors **/
  float aux;
  for (int i = 0; i < numElements; ++i)
  {
    fscanf(input_rawA,"%f",&aux);
    host_A[i] = aux;
    fscanf(input_rawB,"%f",&aux);
    host_B[i] = aux;
  }

  /** Allocate the device input vectors **/
  float *device_A = NULL;
  if (cudaMalloc((void **)&device_A, size) != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device vector A!\n");
    exit(EXIT_FAILURE);
  }

  float *device_B = NULL;
  if (cudaMalloc((void **)&device_B, size) != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device vector B!\n");
    exit(EXIT_FAILURE);
  }

  float *device_C = NULL;
  if (cudaMalloc((void **)&device_C, size) != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device vector C!\n");
    exit(EXIT_FAILURE);
  }

  double start_time = omp_get_wtime();

  /** Copy the host input vectors to device memory **/
  if (cudaMemcpy(device_A, host_A, size, cudaMemcpyHostToDevice) != cudaSuccess)
  {
    fprintf(stderr, "Failed to copy vector A from host to device!\n");
    exit(EXIT_FAILURE);
  }

  if (cudaMemcpy(device_B, host_B, size, cudaMemcpyHostToDevice) != cudaSuccess)
  {
    fprintf(stderr, "Failed to copy vector B from host to device!\n");
    exit(EXIT_FAILURE);
  }

  /** Kernel launch **/
  int threadsPerBlock = 256;
  int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
  vectorMul<<<blocksPerGrid, threadsPerBlock>>>(device_A, device_B, device_C, numElements);

  err = cudaGetLastError();

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to launch vectorMul kernel (error %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  /** Copy the device result vector to host memory. **/
  printf("Copy output data from the CUDA device to the host memory\n");
  if (cudaMemcpy(host_C, device_C, size, cudaMemcpyDeviceToHost) != cudaSuccess)
  {
    fprintf(stderr, "Failed to copy vector C from device to host!\n");
    exit(EXIT_FAILURE);
  }

  double time = omp_get_wtime() - start_time;

  /** Free device global memory **/
  if (cudaFree(device_A) != cudaSuccess)
  {
    fprintf(stderr, "Failed to free device vector A!\n");
    exit(EXIT_FAILURE);
  }

  if (cudaFree(device_B) != cudaSuccess)
  {
    fprintf(stderr, "Failed to free device vector B!\n");
    exit(EXIT_FAILURE);
  }

  if (cudaFree(device_C) != cudaSuccess)
  {
    fprintf(stderr, "Failed to free device vector C!\n");
    exit(EXIT_FAILURE);
  }

  /** Save result vector in output file **/
  FILE * output_C;
  if( ( output_C = fopen( "../input/output.raw", "w" ) ) == NULL )
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

  printf("Done\nProcessing time: %.9lf\n", time);
  return 0;
}
