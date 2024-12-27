![Version](https://img.shields.io/badge/CrntVersion-02.08-dc00ff)
![Author](https://img.shields.io/badge/Author-Falsarolo_Leonardo-6800ff)
![Languages](https://img.shields.io/badge/Languages-C++-0070ff)
![Languages](https://img.shields.io/badge/Languages-Python-00ffd4)
![About](https://img.shields.io/badge/About-Matrix_transposition-lightblue)


**#C++ #Python #Benchmark #Graph #Matrix_transposition #OpenMP #Compile_option #Data**

# Parallel-computing-second-mid-term


## Description
- The goal of this research is to optimize matrix transposition by analyzing the effectiveness of different techniques, focusing on serial execution, OpenMP, and MPI. The serial implementation serves as the baseline method without any form of parallelization and is used as a reference for comparison. OpenMP is employed to enable explicit parallelization within shared memory systems, leveraging specific directives to distribute the workload among threads efficiently. MPI (Message Passing Interface) is used to explore distributed parallelization, where the workload is partitioned across multiple processes that communicate via message passing, making it suitable for clusters or multi-node systems. Through this comparative analysis, the research aims to identify the most efficient approach for enhancing matrix transposition performance across different hardware configurations and parallelization paradigms.
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

- Python
- python libs (pip install)
    * sys
    * matplotlib
    * numpy 
    * csv
    * pandas 
    * from collections import defaultdict
---
## How To Use

To clone and run this application, you'll need [Git](https://git-scm.com) installed on your computer. From your command line:

```bash
# Clone this repository
$ git clone https://github.com/FlsLeonardo/Parallel-computing-mid-term/tree/main

# Go into the repository
$ cd Parallel-computing-mid-term
```
# Using Qsub 
The program will do all the possible matrix transpositions and methods with all different pre setted compilation flags. 
Starting from inside the Parallel-computing-mid-term folder:
```cmd
> qsub -I -q short_cpuQ -l select=1:ncpus=96:ompthreads=96:mem=1mb
> cd method1
> qsub matTranspose.pbs 
---------- wait end execution ----------
> cd bin
> pyton graph.py                    #(for graphs only otherwise do not execute)
---------- wait end execution ----------

> cd ../..
> cd method2
> qsub matTranspose2.pbs 
---------- wait end execution ----------
> cd bin
> pyton graph.py                    #(for graphs only otherwise do not execute)
---------- wait end execution ----------

> cd ../..
> cd method3
> qsub matTranspose3.pbs 
---------- wait end execution ----------
> cd bin
> pyton graph.py                    #(for graphs only otherwise do not execute)
---------- wait end execution ----------

cd ../../best
> pyton best_Implicit_for_matrix_dimension.py
---------- wait end execution ----------
```

The output of the program you can find it in the pbs folder inside the method you chose for example ```method1/pbs```

# Using interactive session

## Example of execution (Cli , FULL)
The full execution will do all the possible matrix transpositions and methods with all different pre setted compilation flags. 

Once downloaded the folder and opend it in your CMD.
```python
> qsub -I -q short_cpuQ -l select=1:ncpus=96:ompthreads=96:mem=1mb
> cd method1/bin
> ./compile.sh   #(if you do not have permission use the chmod command)
---------- wait end execution ----------
> pyton graph.py
---------- wait end execution ----------

> cd ../..
> cd method2/bin
> ./compile.sh
---------- wait end execution ----------
> pyton graph.py
---------- wait end execution ----------

> cd ../..
> cd method3/bin
> ./compile.sh
---------- wait end execution ----------
> pyton graph.py
---------- wait end execution ----------

cd ../../best
> pyton best_Implicit_for_matrix_dimension.py
---------- wait end execution ----------
```
### Compiler.sh parameters
`compiler.sh` has different execution strategies also..
```bash
    Compile.sh $1 $2
    # $1 Compiler Option
    # $2 Matrix Dimension Option
```

- Possiility of selecting one or more Compiler Option 
    - example: ``` > Compile.sh "O1 -funroll-loops" ```
- Possiility of choosing the Compiler Option and the matrix size if you wonnna do a sigle matrix transposition execution with a specific compiler option
    - example: ``` > Compile.sh "O1 -funroll-loops" 10 ```

(_B default the compiler.sh will done the Full execution_)


## Example of execution (CLI, only for a specific type of compiler Option)
Chose one method from:
- **method1** Serial Approach Matrix transposition with 2 for loop nested each startin from 0 to n (where n is the size of the matrix)
- **method2** Optimized Serial Approach Matrix transposition with Block-based Transposition  
- **method3** Optimized Serial Approach Matrix transposition with Diagonal Approach

Then go to the correct folder via `CLI`  (from the main folder)
```cmd
Parallel-computing-mid-term>  cd "Method_chosen"/bin
```
Execute the following command 
```python
Parallel-computing-mid-term/"Method_chosen"/bin>  ./Compile.sh "O2"
```
Or for more Compilation flag
```python
Parallel-computing-mid-term/"Method_chosen"/bin>  ./Compile.sh "O2 -funroll-loops"
```
- Is it possible also to execute `python graph.py` that does all the graphs thanks to the `files.csv` in the output folder
### Graph.py parameters
list of command executable for `graph.py`:
- `--help` all information for the execution
- `-type=("serial,"implicit","omp")` does the graph for only one file.csv ("serial,"implicit","omp")
- `-ES` does the **efficency** and **speedup** graphs for the OpenMP parallelization

(_By default the file graph.py does all the file and all the possible functions_)

## Example of execution (CLI with compiler Option and given matrix size)
Chose one method from:
- **method1** Serial Approach Matrix transposition with 2 for loop nested each startin from 0 to n (where n is the size of the matrix)
- **method2** Optimized Serial Approach Matrix transposition with Block-based Transposition  
- **method3** Optimized Serial Approach Matrix transposition with Diagonal Approach

Then go to the correct folder via `CLI`  (from the main folder)
```cmd
Parallel-computing-mid-term>  cd "Method_chosen"/bin
```
Execute the following command 
```python
Parallel-computing-mid-term/"Method_chosen"/bin>  ./Compile.sh "O2" 10
```
Or for more Compilation flag
```python
Parallel-computing-mid-term/"Method_chosen"/bin>  ./Compile.sh "O2 -funroll-loops" 10
```
- Is it possible also to execute `python graph.py` that does all the graphs thanks to the `files.csv` in the output folder

### Structure of the project
- **Parallel-computing-mid-term**
  - **method1**
    - **bin**
      - main.cpp
      - serialTransposition.cpp
      - implicitTransposition.cpp
      - ompTransposition.cpp
      - Functions.h
      - compile.sh
    - **output**
      - Serial.csv
      - Implicit.csv
      - Omp.csv
    - **pbs**
      - matrix_transpose.o
      - matrix_transpose.e
    - **pbs_graph**
      - contains all the graphs
    - matTranspose.pbs
  - **method2**
    - **bin**
      - main.cpp
      - serialTransposition.cpp
      - implicitTransposition.cpp
      - ompTransposition.cpp
      - Functions.h
      - compile.sh
    - **output**
      - Serial.csv
      - Implicit.csv
      - Omp.csv
    - **pbs**
      - matrix_transpose.o
      - matrix_transpose.e
    - **pbs_graph**
      - contains all the graphs
    - matTranspose2.pbs
  - **method3**
    - **bin**
      - main.cpp
      - serialTransposition.cpp
      - implicitTransposition.cpp
      - ompTransposition.cpp
      - Functions.h
      - compile.sh
    - **output**
      - Serial.csv
      - Implicit.csv
      - Omp.csv
    - **pbs**
      - matrix_transpose.o
      - matrix_transpose.e
    - **pbs_graph**
      - contains all the graphs
    - matTranspose3.pbs
  - **best**  
    - best_Implicit_for_matrix_dimension.py
    - best.csv

For every method there is a brief explanation of the files:

| Directory          | File                  | Descrizione                          |
|---------------------|-----------------------|--------------------------------------|
| `bin/`             | `main.cpp`           | Programma principale                |
|                     | `serialTransposition.cpp` | Trasposizione seriale          |
|                     | `implicitTransposition.cpp` | Trasposizione implicita        |
|                     | `ompTransposition.cpp` | Trasposizione con OpenMP         |
|                     | `Functions.h`        | Header con funzioni condivise       |
|                     | `Main`               | Eseguibile compilato                |
|                     | `compile.sh`         | Script per la compilazione  
|                     | `graph.py`         | Scritp for doing graphs        |
| `output/`          | `Serial.csv`         | Dati trasposizione seriale          |
|                     | `Implicit.csv`       | Dati trasposizione implicita        |
|                     | `Omp.csv`            | Dati trasposizione con OpenMP       |
  