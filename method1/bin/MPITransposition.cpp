#include "Functions.h"
#include <iostream>
#include <vector>
#include <mpi.h>
using namespace std;

void matTransposeMPI(vector<vector<float>>& M,int n,vector<vector<float>>& T,int n_thread){
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
             T[j][i] = M[i][j];
        }
    }
}

bool checkSymMPI(const vector<vector<float>>& M,int n,int n_thread){
    bool isSymmetric = true; 

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (M[i][j] != M[j][i]) {
                isSymmetric = false;
            }
        }
    }

    return isSymmetric;
}

void matTransposeMPI(const std::vector<float>& flatMatrix, std::vector<float>& transposedMatrix, int matSize, int rank, int numProc) {
    // Calcola il numero di righe che ogni processo deve gestire
    int rowsXProcess = matSize / numProc;

    // Local buffer for each process
    std::vector<float> localBlock(rowsXProcess * matSize);
    MPI_Scatter(flatMatrix.data(), rowsXProcess * matSize, MPI_FLOAT,localBlock.data(), rowsXProcess * matSize, MPI_FLOAT,0, MPI_COMM_WORLD);

    // Create a shared memory window for the global transposed matrix
    MPI_Win win;
    float* globalTransposedMatrix = nullptr;

    if (rank == 0) {
        MPI_Win_allocate(matSize * matSize * sizeof(float), sizeof(float),MPI_INFO_NULL, MPI_COMM_WORLD, &globalTransposedMatrix, &win);
    } else {
        MPI_Win_allocate(0, sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD,&globalTransposedMatrix, &win);
    }

    MPI_Win_fence(0, win);

    // Each process transposes its local block and writes into the global matrix
    for (int i = 0; i < rowsXProcess; ++i) {
        for (int j = 0; j < matSize; ++j) {
            int globalRow = j;
            int globalCol = rank * rowsXProcess + i;
            float value = localBlock[i * matSize + j];

            MPI_Put(&value, 1, MPI_FLOAT, 0, globalRow * matSize + globalCol,1, MPI_FLOAT, win);
        }
    }

    MPI_Win_fence(0, win);

    // Rank 0 saves the transposed matrix
    if (rank == 0) {
        transposedMatrix = std::vector<float>(globalTransposedMatrix,globalTransposedMatrix + matSize * matSize);
    }

    MPI_Win_free(&win);
}