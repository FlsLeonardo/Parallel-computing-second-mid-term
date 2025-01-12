![Version](https://img.shields.io/badge/CrntVersion-4.11-dc00ff)
![Author](https://img.shields.io/badge/Author-Falsarolo_Leonardo-6800ff)
![Languages](https://img.shields.io/badge/Languages-C++-0070ff)
![Languages](https://img.shields.io/badge/Languages-Python-00ffd4)
![About](https://img.shields.io/badge/About-Matrix_transposition-lightblue)


**#C++ #Python #Benchmark #Graph #Matrix_transposition #OpenMP #Mpi #Mpirun #mpiprocs #mpicxx #Data**

<div style="position: relative; display: inline-block; text-align: center;">
  <img src="https://images.unsplash.com/photo-1667372459510-55b5e2087cd0?q=80&w=1932&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" alt="Beautiful Landscape" width="100%">
  <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: white; font-size: 24px; background-color: rgba(0, 0, 0, 0.5); padding: 10px;">
  <h3>Parallel computing<h3>
  </div>
</div>

## Description of the Second Mid Term
The goal of this research is to optimize matrix transposition by analyzing the effectiveness of different techniques, focusing on serial execution, OpenMP, and MPI. The serial implementation serves as the baseline method without any form of parallelization and is used as a reference for comparison. OpenMP is employed to enable explicit parallelization within shared memory systems, leveraging specific directives to distribute the workload among threads efficiently. MPI (Message Passing Interface) is used to explore distributed parallelization, where the workload is partitioned across multiple processes that communicate via message passing, making it suitable for clusters or multi-node systems. 

Through this comparative analysis, the research aims to identify the most efficient approach for enhancing matrix transposition performance across different hardware configurations and parallelization paradigms. Additionally, the study investigates the following key aspects:

- **Efficiency**: Evaluating the overall performance of each implementation.
- **Speedup**: Analyzing the performance gains achieved through parallelization.
- **Bandwidth**: Assessing the communication overhead and data transfer rates in each method.

This study provides valuable insights into the performance scaling and communication overhead associated with serial, OpenMP, and MPI implementations of matrix transposition.

---
## Requirements

- C++
- C++ libs
    * iostream
    * vector 
    * cstdlib 
    * ctime 
    * cmath
    * random
    * omp.h
    * fstream
    * mpi.h

- Python
- python libs (pip install)
    * matplotlib
    * numpy 
    * pandas 
---
## How To Use

To clone and run this application, you'll need [Git](https://git-scm.com) installed on your computer. From your command line:

```bash
# Clone this repository
$ git clone git@github.com:FlsLeonardo/Parallel-computing-second-mid-term.git

# Go into the repository
$ cd Parallel-computing-second-mid-term
```
# Using Qsub 
The program will do automatically some of the possible matrix transpositions using OMP parallelizzation and MPI parallelizzation. 
Starting from inside the Parallel-computing-mid-term folder:
```cmd
> qsub -I -q short_cpuQ -l select=1:ncpus=96:ompthreads=96:mem=1mb
> cd method1
> qsub matTranspose.pbs 
---------- wait end execution ----------
> cd py
> pyton graph.py                    #(for graphs only otherwise do not execute)
---------- wait end execution ----------
```

The output of the program you can find it in the pbs folder ```method1/pbs```

# Using interactive session

## Example of execution (Cli , FULL)
The full execution will do all the possible matrix transpositions and methods with all different pre setted compilation flags. 

Once downloaded the folder and opend it in your CMD.
```python
> qsub -I -q short_cpuQ -l select=1:ncpus=64:mpiprocs=64:mem=256mb
> cd method1/bin
> ./compile.sh   #(if you do not have permission use the chmod command)
---------- wait end execution ----------
> pyton graph.py
---------- wait end execution ----------
```
### Compiler.sh
Without parameters it compiles only the files that there are in the bin directory. 
So you do not haave to compile every file every time.

### Compiler.sh parameters
With parmeters it compile and run the project.
`compiler.sh` has different execution strategies also..
```bash
    Compile.sh $1 $2
    # $1 Number of Processors used
    # $2 Size of the Matrix
```

- Possiility of executing with 1 parameter: 
    - example: ``` > Compile.sh 16 ``` witch are the number of Processors used
- Possiility of choosing 2 parameters:
    - example: ``` > Compile.sh 16 10 ``` so 16 Processors used to compute the transposition of a 2^20 x 2^10 matrix

(_By default the compiler.sh will only compiles the file_)

## Using Mpirun
We have 3 possibilities of execution:

1) Choosing only the number of Processors and then doing everything ```(Seqential code , Omp code , MPI code)```.
```python
mpirun -np "number_of_Processors" ./Main  
```
2) Choosing the number of Processors and then doing only the MPI code for that given ```Matrix Size```.
```python
mpirun -np "number_of_Processors" ./Main  mpi "matrix_size"
```
3) Choosing the number of Processors and then doing ```(Serial,OMP,MPI)``` codes for that given ```Matrix Size```.
  ```python
  mpirun -np "number_of_Processors" ./Main  all "matrix_size"
  ```


### Example of execution with Mpirun (CLI, only for a specific Size and MPI number of Procs)

Go to the correct folder via `CLI`  (from the main folder)
```cmd
Parallel-computing-mid-term>  cd "Method1"/bin
```
Execute the following command 
```python
Parallel-computing-mid-term/"Method1"/bin>  ./Compile.sh 
```
Than execute the Main program:
```python
Parallel-computing-mid-term/"Method1"/bin>  mpirun -np "number_of_Processors" ./Main  all "matrix_size"
```

- Is it possible also to execute `python graph.py` that does all the graphs thanks to the `files.csv` in the output folder
### Graph.py parameters
list of command executable for `graph.py`:
- `--help` all information for the execution
- `-type=("Bandwidth,efficency,speedup")` does the graph of 1 of theese 3 options

(_By default the file graph.py does all the file and all the possible functions_)

### Structure of the project
- **Parallel-computing-Second-mid-term**
  - **method1**
    - **bin**
      - main.cpp
      - serialTransposition.cpp
      - MPITransposition.cpp
      - ompTransposition.cpp
      - Functions.h
      - compile.sh
    - **output**
      - Serial.csv
      - MPI.csv
      - Omp.csv
    - **pbs**
      - matrix_transpose.o
      - matrix_transpose.e
    - **pbs_graph**
      - contains all the graphs
    -**py**
      -graphs.py
    - matTranspose.pbs

For every method there is a brief explanation of the files:

| Directory          | File                  | Descrizione                          |
|---------------------|-----------------------|--------------------------------------|
| `bin/`             | `main.cpp`           | main program                   |
|                     | `serialTransposition.cpp` | serial trnspose          |
|                     | `MPITransposition.cpp` | mpi transpose               |
|                     | `ompTransposition.cpp` | omp transpose               |
|                     | `Functions.h`        | shared functions across files |
|                     | `Main`               | Eseguibiexecutable compiled   |
|                     | `compile.sh`         | Script for the compilation    | 
| `output/`          | `Serial.csv`         | Data of serial transpose       |
|                     | `MPI.csv`       | Dati of mpi trnapsoe               |
|                     | `Omp.csv`            | Dati of omp transpose         |
| `py/` 
|                     | `graphs.py`         | Scritp for doing graphs        |