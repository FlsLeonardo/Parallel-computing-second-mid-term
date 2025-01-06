#include "Functions.h"
#include <iostream>
#include <vector>
#include <mpi.h>
using namespace std;

int checkSymMPI(const std::vector<float>& mat, int n, int mpi_rank, int mpi_size){
    int workers = mpi_size;
    int rows_per_worker = n / workers;
    int remaining_rows = n % workers;

    int start_row = (mpi_rank) * rows_per_worker + std::min(mpi_rank, remaining_rows);
    int end_row = start_row + rows_per_worker + (mpi_rank < remaining_rows);

    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < n; ++j) {
            if (mat[i * n + j] != mat[j * n + i]) return 0;
        }
    }
    return 1;
}

void matTransposeMPI2(const std::vector<float>& mat,std::vector<float>& trans, int n, int mpi_rank, int mpi_size){
    int workers = mpi_size;
    int rows_per_worker = n / workers;
    int remaining_rows = n % workers;

    int start_row = (mpi_rank) * rows_per_worker + std::min(mpi_rank, remaining_rows);
    int end_row = start_row + rows_per_worker + (mpi_rank < remaining_rows);

    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < n; ++j) {
            trans[i * n + j] = mat[j * n + i];
        }
    }
    MPI_Allgather(MPI_IN_PLACE,(end_row-start_row)*n,MPI_FLOAT,trans.data(),(end_row-start_row)*n,MPI_FLOAT,MPI_COMM_WORLD);
}

void matTransposeMPI3(const std::vector<float>& mat, std::vector<float>& trans, int n, int mpi_rank, int mpi_size) {
    int workers = mpi_size;
    int rows_per_worker = n / workers;
    int remaining_rows = n % workers;

    // Calcolo dei range di righe per ogni rank
    int start_row = mpi_rank * rows_per_worker + std::min(mpi_rank, remaining_rows);
    int end_row = start_row + rows_per_worker + (mpi_rank < remaining_rows);
    int local_rows = end_row - start_row;

    // Ogni rank trasforma solo le proprie righe assegnate
    std::vector<float> local_trans(local_rows * n);
    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < n; ++j) {
            local_trans[(i - start_row) * n + j] = mat[j * n + i];
        }
    }

    // Raccogli i risultati su rank 0
    MPI_Gather(
        local_trans.data(),         // Buffer locale
        local_rows * n,             // Numero di elementi da inviare
        MPI_FLOAT,                  // Tipo di dati
        trans.data(),               // Buffer di raccolta (solo su rank 0)
        local_rows * n,             // Numero di elementi da ricevere per processo
        MPI_FLOAT,                  // Tipo di dati
        0,                          // Destinatario (rank 0)
        MPI_COMM_WORLD              // Comunicatore
    );

    // Ora solo il rank 0 possiede la matrice completa
}

void matTransposeMPI4(const std::vector<float>& mat, std::vector<float>& trans, int n, int mpi_rank, int mpi_size) { //coc implementatios not use it
    int workers = mpi_size;
    int rows_per_worker = n / workers;
    int remaining_rows = n % workers;

    // Calculate start and end rows for this process
    int start_row = mpi_rank * rows_per_worker + std::min(mpi_rank, remaining_rows);
    int end_row = start_row + rows_per_worker + (mpi_rank < remaining_rows);

    // Local buffer for the transposed rows this rank computes
    int local_rows = end_row - start_row;
    std::vector<float> local_trans(local_rows * n);

    // Perform the local transposition
    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < n; ++j) {
            local_trans[(i - start_row) * n + j] = mat[j * n + i];
        }
    }

    // Gather the transposed parts to rank 0
    MPI_Gather(
            local_trans.data(),           // Send buffer (local transposed part)
            local_rows * n,               // Number of elements to send
            MPI_FLOAT,                    // Data type
            trans.data(),                 // Receive buffer on rank 0 (final transposed matrix)
            local_rows * n,               // Number of elements each rank sends
            MPI_FLOAT,                    // Data type
            0,                            // Root rank
            MPI_COMM_WORLD                // Communicator
    );
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