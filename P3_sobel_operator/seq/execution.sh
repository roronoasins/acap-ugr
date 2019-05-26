#!/bin/sh
#cd /home/user/path_to_the_program/


for file in input/*; do
  for i in $(seq 0 3)
    do
      ./sobel_def $file
    done
done
