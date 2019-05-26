#!/bin/sh

for file in ../input/*; do
  for j in $(seq 0 3)
  do
    ./a.out $file
  done
done
