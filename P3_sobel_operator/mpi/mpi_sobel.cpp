// > compile with mpic++ mpi_sobel.cpp  -o mpi_sobel `pkg-config --libs opencv` -fopenmp -lstdc++
#include <iostream>
#include <cmath>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <omp.h>
#include <mpi.h>
#include <fstream>

const int MAXBYTES=512*1024*1024;
uchar buffer[MAXBYTES];

using namespace std;
using namespace cv;

/** Source and destiny images **/
Mat src, dst;

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
int x_gradient(Mat image, int x, int y)
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

/**
 * Pack and send m to dest
 * @param m    [image]
 * @param dest [destiny]
 */
void matsnd(const Mat& m,int dest){
  int rows  = m.rows;
  int cols  = m.cols;
  int type  = m.type();
  int channels = m.channels();
  memcpy(&buffer[0 * sizeof(int)],(uchar*)&rows,sizeof(int));
  memcpy(&buffer[1 * sizeof(int)],(uchar*)&cols,sizeof(int));
  memcpy(&buffer[2 * sizeof(int)],(uchar*)&type,sizeof(int));

  // Find the size of each matrix row in bytes, and multiply it by the number of rows
  size_t sizeInBytes = m.step[0] * m.rows;

  if(!m.isContinuous())
  {
    m.copyTo(m);
    cout << "\t **** no-continuous ****" << endl;
  }
  memcpy(&buffer[3*sizeof(int)],m.data,sizeInBytes);
  MPI_Send(&buffer,sizeInBytes+3*sizeof(int),MPI_UNSIGNED_CHAR,dest,0,MPI_COMM_WORLD);
  //cout << "sent to " << dest <<  endl;
}

/**
 * Unpack and recieve image from src
 * @param  src    [source]
 * @param  status [actual status]
 * @return        [image recieved]
 */
Mat matrcv(int src, MPI_Status& status){
  int count,rows,cols,type,channels;

  MPI_Recv(&buffer,sizeof(buffer),MPI_UNSIGNED_CHAR,src,0,MPI_COMM_WORLD,&status);
  //cout << "recieved by " << src <<  endl;
  MPI_Get_count(&status,MPI_UNSIGNED_CHAR,&count);
  memcpy((uchar*)&rows,&buffer[0 * sizeof(int)], sizeof(int));
  memcpy((uchar*)&cols,&buffer[1 * sizeof(int)], sizeof(int));
  memcpy((uchar*)&type,&buffer[2 * sizeof(int)], sizeof(int));

  Mat received= Mat(rows,cols,type,(uchar*)&buffer[3*sizeof(int)]);
  return received;
}

int main(int argc, char** argv)
{
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

  Mat img;
  int rows_av, rows_extra;
  Size s = src.size();
  int rows = s.height;
  int cols = s.width;
  int type = src.type();
  uchar pic[rows*cols];
  int pic_struct[3];
  int np, ip;
  double start_time = omp_get_wtime();
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS){
    exit(1);
  }
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &ip);
  MPI_Status status;
  if(ip==0)
  {
    // rows_limit and new_rows are used to distinguish between top, bottom and middle cases
    // and could use sobel operator to all pixels without memory access issues
    int rows_limit, new_rows;
    for(int i=1; i < np; i++)
    {
      if (i < np-1)
      {
        rows_limit = 1;
        new_rows = 2;
      }else
        {
          rows_limit = 1;
          new_rows = 1;
        }
      //cout << "sending to " << i << endl;
      rows_av = rows/np;
      pic_struct[0] = new_rows;
      pic_struct[1] = cols;
      pic_struct[2] = rows_av;
      MPI_Send(&pic_struct, sizeof(pic_struct), MPI_BYTE, i, 0, MPI_COMM_WORLD);
     /*
      * Need to send just the piece of picture each rank needs
      *
      * Using ptr to src because Mat constructor does not copy the picture,
      * just use the pointer to, more efficent than copy all data
      */
      matsnd(Mat(Size(cols,rows_av+new_rows),type,src.ptr<uchar>(i*rows_av-rows_limit)), i);
    }
    rows_extra = 1;
    img = Mat(Size(cols,rows_av+rows_extra),type,src.ptr<uchar>(0));
  }else{//ip
    MPI_Recv(&pic_struct, sizeof(pic_struct), MPI_BYTE, 0, 0, MPI_COMM_WORLD, &status);
    img = matrcv(0, status);
  }
int extra;
  if(ip)
  {
    rows_extra = pic_struct[0];
    cols = pic_struct[1];
    rows_av = pic_struct[2];
  }
  // last rank has just 1 new row but it needs to start in row=1 not row=0(as all ranks!=0 do)
  if (ip == np-1) ++rows_extra;
  uchar picAux[rows_av*cols];
  int ip_gx, ip_gy, sum;
  //cout << "---- loop " << ip << endl;
  for(int y = 0; y < rows_av; y++){
    for(int x = 0; x < cols  ; x++){
      ip_gx = x_gradient(img, x, y+rows_extra-1);
      ip_gy = y_gradient(img, x, y+rows_extra-1);
      sum = abs(ip_gx) + abs(ip_gy);
      sum = sum > 255 ? 255:sum;
      sum = sum < 0 ? 0 : sum;
      picAux[y*cols+x] = sum;
    }
  }
  cout << endl;
  //cout << "---- loop ends " << ip << endl;
  MPI_Gather(picAux, cols*rows_av, MPI_UNSIGNED_CHAR, pic, cols*rows_av, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
  //cout << "gather" << ip << endl;
  MPI_Finalize();
  //cout << "finalize" << ip << endl;


  if(!ip)
  {
    double time = omp_get_wtime() - start_time;
    cout << "\nNumber of processes: " << np << endl;
    cout << "Rows, Cols: " << rows << " " << cols << endl;
    cout << "Rows, Cols(Division): " << rows_av << ", " << cols << endl << endl;

    /** Get output file string and saves it **/
    cout << "Processing time: " << time << endl;
    for(int i=0; i < 6 ; i++)  picture.erase(picture.begin());
    for(int i=0; i < 4 ; i++)  picture.pop_back();
    picture.insert(0,"output/");
    picture += "-sobel.jpg";

    src.release(); //free src
    dst.create(rows, cols, type);
    for(int y = 0; y < rows; y++)
      for(int x = 0; x < cols; x++)
        dst.at<uchar>(y,x) = pic[y*cols+x];

    /** Save output image **/
    if(imwrite(picture.c_str(), dst)) cout << "Picture correctly saved as " << picture << endl;
    else  cout << "\nError has occurred being saved." << endl;
    dst.release(); //free dst

    /** Outpute file used for stadistics **/
    fstream data;
    data.open("data.txt", ofstream::out | ofstream::ate | ofstream::app);
    data << np << " Processes\tPicture: " << rows*cols << " rows:" << rows << "cols: " << cols << endl << "Proc time: " << time << " -> " << picture << endl << endl;
    data.close();
  }

  return 0;
}
