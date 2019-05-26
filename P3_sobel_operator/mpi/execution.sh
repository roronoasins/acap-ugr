#!/bin/sh

for file in input/*; do
  for i in $(seq 2 4)
  do
    for j in $(seq 0 3)
    do
      mpirun -np $i mpi_sobel $file
    done
  done
done
