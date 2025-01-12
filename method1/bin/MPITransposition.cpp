#include "Functions.h"
#include <iostream>
#include <vector>
#include <mpi.h>
using namespace std;

int checkSymMPI(const std::vector<float>& mat, int n, int mpi_rank, int mpi_size){
    int works = mpi_size;
    int rows_x_worker = n / works;
    int remain_rows = n % works;

    int start_row = (mpi_rank) * rows_x_worker + std::min(mpi_rank, remain_rows);
    int end_row = start_row + rows_x_worker + (mpi_rank < remain_rows);

    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < n; ++j) {
            if (mat[i * n + j] != mat[j * n + i]) return 0;
        }
    }
    return 1;
}


void matTransposeMPI(const std::vector<float>& mat, std::vector<float>& l_trans, int n, int mpi_rank, int start_row ,int end_row) {

    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < n; ++j) {
            l_trans[(i - start_row) * n + j] = mat[j * n + i];
        }
    }
}