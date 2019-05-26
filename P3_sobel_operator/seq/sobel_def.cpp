#include <iostream>
#include <cmath>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <omp.h>
#include <fstream>

using namespace std;
using namespace cv;

//mpicc sobel.cpp  -o sobel `pkg-config --libs opencv` -fopenmp -lstdc++

/**
 * Computes the x component of the gradient vector
 * at a given point in a image.
 *          | 1 0 -1 |
 *     Gx = | 2 0 -2 |
 *          | 1 0 -1 |
 *
 * @param  image [src image]
 * @param  x     [coordinate x]
 * @param  y     [coordinate y]
 * @return       [gradient in the x direction]
 */
int x_gradient(Mat image, int x, int y) //cols, rows
{
  int gradient;
  if (x!=0 && y!=0)
  {
    gradient = image.at<uchar>(y-1, x-1) +
               2*image.at<uchar>(y, x-1) +
               image.at<uchar>(y+1, x-1) -
               image.at<uchar>(y-1, x+1) -
               2*image.at<uchar>(y, x+1) -
               image.at<uchar>(y+1, x+1);
  }else
  {
    if (x==0 && y!=0)
    {
      gradient = image.at<uchar>(y-1, x+1) -
                 2*image.at<uchar>(y, x+1) -
                 image.at<uchar>(y+1, x+1);
    }else if (x=!0 && y==0)
          {
            gradient = 2*image.at<uchar>(y, x-1) +
                       image.at<uchar>(y+1, x-1) -
                       2*image.at<uchar>(y, x+1) -
                       image.at<uchar>(y+1, x+1);
          }else if (x=!0 && y==image.cols)
                {
                  gradient = image.at<uchar>(y-1, x-1) +
                             2*image.at<uchar>(y, x-1) +
                             image.at<uchar>(y-1, x+1) -
                             2*image.at<uchar>(y, x+1);
                }
  }

  return gradient;
}

/**
 * Computes the y component of the gradient vector
 * at a given point in a image.
 *          | 1  2  1 |
 *     Gy = | 0  0  0 |
 *          |-1 -2 -1 |
 *
 * @param  image [src image]
 * @param  x     [pixel x]
 * @param  y     [pixel y]
 * @return       [gradient in the y direction]
 */
int y_gradient(Mat image, int x, int y)
{
  int gradient;
  if (x!=0 && y!=0)
  {
    gradient = image.at<uchar>(y+1, x-1) +
               2*image.at<uchar>(y+1, x) +
               image.at<uchar>(y+1, x+1) -
               image.at<uchar>(y-1, x-1) -
               2*image.at<uchar>(y-1, x) -
               image.at<uchar>(y-1, x+1);
  }else
  {
    if (x==0 && y!=0)
    {
      gradient = 2*image.at<uchar>(y+1, x) +
                 image.at<uchar>(y+1, x+1) -
                 2*image.at<uchar>(y-1, x) -
                 image.at<uchar>(y-1, x+1);
    }else if (x=!0 && y==0)
          {
            gradient = image.at<uchar>(y+1, x-1) +
                       2*image.at<uchar>(y+1, x) +
                       image.at<uchar>(y+1, x+1);
          }else if (x=!0 && y==image.cols)
                {
                  gradient = image.at<uchar>(y-1, x-1) -
                             2*image.at<uchar>(y-1, x) -
                             image.at<uchar>(y-1, x+1);
                }
  }

  return gradient;
}

int main(int argc, char** argv)
{

  Mat src, dst;
  int gx, gy, sum;
  string picture;
  if (argc == 2) {
    picture = argv[1];
    src = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
  }
  else {
    picture = "input/logan.jpg";
    src = imread(picture.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
  }

  if( !src.data )
  { return -1; }
  dst.create(src.rows, src.cols, src.type());

  double start_time = omp_get_wtime();
  for(int y = 0; y < src.rows; y++){
    for(int x = 0; x < src.cols; x++){
      gx = x_gradient(src, x, y);
      gy = y_gradient(src, x, y);
      sum = abs(gx) + abs(gy);
      sum = sum > 255 ? 255:sum;
      sum = sum < 0 ? 0 : sum;
      dst.at<uchar>(y,x) = sum;
    }
  }
  double time = omp_get_wtime() - start_time;

  /** Get output file string and saves it **/
  cout << "Processing time: " << time << endl;
  for(int i=0; i < 6 ; i++)  picture.erase(picture.begin());
  for(int i=0; i < 4 ; i++)  picture.pop_back();
  picture.insert(0,"output/");
  picture += "-sobel.jpg";

  /** Save output image **/
  if(imwrite(picture.c_str(), dst)) cout << "Picture correctly saved as " << picture << endl;
  else  cout << "\nError has occurred being saved." << endl;

  /** Outpute file used for stadistics **/
  fstream data;
  data.open("data.txt", ofstream::out | ofstream::ate | ofstream::app);
  data << "Picture: " << src.rows*src.cols << " rows: " << src.rows << "cols: " << src.cols << endl << "Proc time: " << time << " -> " << picture << endl << endl;
  data.close();
  return 0;
}
