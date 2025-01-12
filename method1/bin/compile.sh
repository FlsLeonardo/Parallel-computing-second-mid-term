#!/bin/bash

if [ $# -eq 0 ]; then
  module load mpich-3.2.1--gcc-9.1.0
  
  
  mpicxx -c main.cpp -fopenmp
  mpicxx -c serialTransposition.cpp
  mpicxx -c MPITransposition.cpp
  mpicxx -c ompTransposition.cpp -fopenmp
  mpicxx main.o serialTransposition.o MPITransposition.o ompTransposition.o -o Main -fopenmp

  
fi

if [ $# -eq 1 ]; then
  module load mpich-3.2.1--gcc-9.1.0
  
  mpicxx -c main.cpp -fopenmp
  mpicxx -c serialTransposition.cpp
  mpicxx -c MPITransposition.cpp 
  mpicxx -c ompTransposition.cpp -fopenmp
  mpicxx main.o serialTransposition.o MPITransposition.o ompTransposition.o -o Main -fopenmp
  mpirun -np $1 ./Main  all 12
fi

if [ $# -eq 2 ]; then
  module load mpich-3.2.1--gcc-9.1.0
  
  mpicxx -c main.cpp -fopenmp
  mpicxx -c serialTransposition.cpp
  mpicxx -c MPITransposition.cpp 
  mpicxx -c ompTransposition.cpp -fopenmp
  mpicxx main.o serialTransposition.o MPITransposition.o ompTransposition.o -o Main -fopenmp
  mpirun -np $1 ./Main  all $2
fi