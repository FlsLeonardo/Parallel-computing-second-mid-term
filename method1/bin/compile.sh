#!/bin/bash

if [ $# -eq 0 ]; then
  module load mpich-3.2.1--gcc-9.1.0
  
  
  mpicxx -c main.cpp -fopenmp
  mpicxx -c serialTransposition.cpp
  mpicxx -c MPITransposition.cpp -O1
  mpicxx -c ompTransposition.cpp -fopenmp
  mpicxx main.o serialTransposition.o MPITransposition.o ompTransposition.o -o Main -fopenmp
  mpirun -np 8 ./Main

  
fi

# Se viene passato un solo parametro (O1, O2, etc.)
if [ $# -eq 1 ]; then
  module load mpich-3.2.1--gcc-9.1.0
  
  mpicxx -c main.cpp -fopenmp
  mpicxx -c serialTransposition.cpp
  mpicxx -c MPITransposition.cpp -O1
  mpicxx -c ompTransposition.cpp -fopenmp
  mpicxx main.o serialTransposition.o MPITransposition.o ompTransposition.o -o Main -fopenmp
  mpirun -np $1 ./Main
fi
