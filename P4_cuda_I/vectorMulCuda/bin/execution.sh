#!/bin/sh
prog_path=""

for j in $(seq 0 10)
do
	for i in $(seq 0 3)
	do
		arg1="../input/input"$j"0.raw"
		arg2="../input/input"$j"1.raw"
		$prog_path/vectorMul $arg1 $arg2
	done
done
