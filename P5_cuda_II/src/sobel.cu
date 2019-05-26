#include <stdlib.h>
#include <stdio.h>
#include <string.h>
/** CUDA utilities and system includes **/
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_functions.h> // includes for SDK helper functions
#include <helper_cuda.h>      // includes for cuda initialization and error checking

/** OpenCV utilities and system includes **/
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

/** OpenMP library, used for processing time **/
#include <omp.h>

/** https://docs.nvidia.com/cuda/profiler-users-guide/index.html#prepare-application	**/
//#include <cuda_profiler_api.h>

using namespace std;
using namespace cv;
// /usr/local/cuda-10.1/bin/nvcc --cudart static --relocatable-device-code=false -gencode arch=compute_50,code=compute_50 -gencode arch=compute_50,code=sm_50 -m64 sobel.cu `pkg-config --libs opencv` -lstdc++ -I/usr/local/cuda-10.1/samples/common/inc/ -I/usr/local/cuda-10.1/include -I/usr/include/opencv/ -I/usr/include/opencv2/
// openmp -> -Xcompiler -fopenmp

#define THREADS_DIM 32.0

 /**
  * [sobel_gpu This function runs on the GPU the sobel filter, on a 2D grid the current gradients to each (i,j) pair are computed.]
  * @param src    [original image]
  * @param gpu    [final image created using the sobel filter]
  * @param width  [width of the image]
  * @param height [height of the image]
  */
__global__ void sobel_gpu(const uchar* src, uchar* gpu, const unsigned int width, const unsigned int height)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int gx, gy, sum;
	if( x > 0 && y > 0 && x < width-1 && y < height-1) {
	  gx = (-1* src[(y-1)*width + (x-1)]) + (-2*src[y*width+(x-1)]) + (-1*src[(y+1)*width+(x-1)]) +
	       (    src[(y-1)*width + (x+1)]) + ( 2*src[y*width+(x+1)]) + (   src[(y+1)*width+(x+1)]);
	  gy = (    src[(y-1)*width + (x-1)]) + ( 2*src[(y-1)*width+x]) + (   src[(y-1)*width+(x+1)]) +
	       (-1* src[(y+1)*width + (x-1)]) + (-2*src[(y+1)*width+x]) + (-1*src[(y+1)*width+(x+1)]);
 		sum = abs(gx) + abs(gy);
 		sum = sum > 255 ? 255:sum;
 		sum = sum < 0 ? 0 : sum;
	  gpu[y*width + x] = sum;
	}
}

int main(int argc, char **argv)
{
	Mat src;
	string picture;
	if (argc == 2) {
	  picture = argv[1];
	  src = imread(picture, CV_LOAD_IMAGE_GRAYSCALE);
	  printf("Image read successfully\n");
	}
	else {
	  picture = "../input/logan.jpg";
	  src = imread(picture, CV_LOAD_IMAGE_GRAYSCALE);
	}
	unsigned int width = src.cols, height = src.rows;

	/** Allocate space in the GPU for our original img, new img, and dimensions **/
	uchar *gpu_orig, *gpu_sobel;
	if (cudaMalloc((void**)&gpu_orig, width*height*sizeof(uchar)) != cudaSuccess)
	{
			fprintf(stderr, "Failed to allocate device source image!\n");
			exit(EXIT_FAILURE);
	}
	if (cudaMalloc((void**)&gpu_sobel, width*height*sizeof(uchar)) != cudaSuccess)
	{
			fprintf(stderr, "Failed to allocate device output image!\n");
			exit(EXIT_FAILURE);
	}

	//cudaProfilerStart();
	double start_time = omp_get_wtime();
	/** Transfer over the memory from host to device and memset the sobel array to 0s **/
	if (cudaMemcpy(gpu_orig, src.data, width*height*sizeof(uchar), cudaMemcpyHostToDevice) != cudaSuccess)
	{
			fprintf(stderr, "Failed to copy source image from host to device!\n");
			exit(EXIT_FAILURE);
	}

	/** set up the dim3's gpu (threads per block and num of blocks (per grid))**/
  dim3 threadsPerBlock(THREADS_DIM, THREADS_DIM, 1);	// THREADS_DIM x THREADS_DIM x 1	-- 32 x 32 = 1024
	/** ceil(x) -> the smallest integer greater than or equal to x **/
	dim3 numBlocks(ceil(width/THREADS_DIM), ceil(height/THREADS_DIM), 1);	// width/GRIDVAL x height/GRIDVAL x 1

	/** Kernel launch **/
	sobel_gpu<<<numBlocks, threadsPerBlock>>>(gpu_orig, gpu_sobel, width, height);

	/** Copy data back to CPU from GPU **/
	uchar *dst = new uchar[height*width];
	/** CPU is blocked until mem is copied, memory copy starts when kernel(sobel_gpu) finishes **/
	if (cudaMemcpy(dst, gpu_sobel, width*height*sizeof(uchar), cudaMemcpyDeviceToHost) != cudaSuccess)
	{
			fprintf(stderr, "Failed to copy output image from device to host!\n");
			exit(EXIT_FAILURE);
	}
	double cuda_time = omp_get_wtime() - start_time;
	//cudaProfilerStop();

	/** Output sobel filtering runtime **/
	printf("\nProcessing %s: %d rows x %d columns\n", argv[1], height, width);
	printf("CUDA execution time   = %*.9f s\n", 5, cuda_time);
	printf("\n");

	/** Get output file string and saves it **/
	for(int i=0; i < 8 ; i++)  picture.erase(picture.begin());
	for(int i=0; i < 4 ; i++)  picture.pop_back();
	picture.insert(0,"../output");
	picture += "-sobel.jpg";

  Mat final(height, width, CV_8U);
	for(int i = 0; i < height; i++)
		for(int j = 0; j < width; j++)
			final.at<uchar>(i,j) = (uchar)dst[i*width+j];

	if(imwrite(picture.c_str(), final)) cout << "Picture correctly saved as " << picture << endl;
  else  cout << "\nError has occurred being saved." << endl;

	/*FILE * data;
	data = fopen("data.txt", "a");
	fprintf(data,"Picture size: %d Proc time:\t%.10f\n\n", height*width, cuda_time);
	fclose(data);*/

	/** Free any memory used **/
	cudaFree(gpu_orig);
	cudaFree(gpu_sobel);
	free(dst);
	src.release();
	final.release();

  return 0;
}
