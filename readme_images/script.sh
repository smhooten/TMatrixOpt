#!/bin/bash
for i in 1 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40
do
   echo $i >> mpi_data2.txt
   mpirun -n $i python lumerical_comparison.py >> mpi_data2.txt
done
