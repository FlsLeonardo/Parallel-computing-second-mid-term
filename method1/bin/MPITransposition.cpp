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

void matTransposeFlattenedMPI(const std::vector<float>& flatMatrix, std::vector<float>& transposedMatrix, int n, int rank, int size) {
    // Calcola il numero di righe che ogni processo deve gestire
    int rowsPerProcess = n / size;
    int remainingRows = n % size;  // Le righe rimanenti per i processi rimanenti
    int startRow = rank * rowsPerProcess;
    int endRow = (rank + 1) * rowsPerProcess;
    
    if (rank == size - 1) {
        endRow += remainingRows;  // L'ultimo processo prende anche le righe rimanenti
    }

    // Prepara un array per la parte della matrice che questo processo gestirà
    std::vector<float> localMatrix(rowsPerProcess * n, 0.0f);
    //std::vector<float> localtransposedMatrix(rowsPerProcess * n, 0.0f);

    // Scatter: distribuiamo le righe della matrice originale tra i processi
    MPI_Scatter(flatMatrix.data(), rowsPerProcess * n, MPI_FLOAT, localMatrix.data(), rowsPerProcess * n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Trasposizione locale: ognuno dei processi prende le righe e le trasforma in colonne
    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < n; ++j) {
            transposedMatrix[j * n + i] = localMatrix[(i - startRow) * n + j]; //localtransposedMatrix
        }
    }
    //MPI_Gather(localtransposedMatrix.data(), rowsPerProcess * n, MPI_FLOAT, transposedMatrix.data(), rowsPerProcess * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
  
    // Ogni processo stampa la propria parte della matrice trasposta
    //std::cout << "Processo " << rank << " ha trasposto le righe:\n";
    //for (int i = startRow; i < endRow; ++i) {
      //  for (int j = 0; j < n; ++j) {
        //    std::cout << transposedMatrix[i * n + j] << " ";
        //}
        //std::cout << std::endl;
    //}
    
    //PER FANE ANDARE METà TOLGO LA GATHER PIU CH LTRO TENGO COMMENTATA LA GATHER E  TOLGO LA LOCALTRANSPOSITIONMATRIX METTENDO LA TRANSPOSITIONMATRIX
}