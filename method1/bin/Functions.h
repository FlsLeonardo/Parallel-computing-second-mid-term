#ifndef FUNCTIONS_H
#define FUNCTIONS_H
#include <vector>
using namespace std;

void matTransposeSerial(vector<vector<float>>& M,int n,vector<vector<float>>& T);      
void matTransposeOmp(vector<vector<float>>& M,int n,vector<vector<float>>& T, int n_thread);
void matTransposeMPI(const vector<float>& flatMatrix, vector<float>& transposedMatrix, int matSize, int rank, int numProc);
void matTransposeMPI2(const std::vector<float>& mat,std::vector<float>& trans, int n, int mpi_rank, int mpi_size);
void matTransposeMPI3(const std::vector<float>& mat, std::vector<float>& trans, int n, int mpi_rank, int mpi_size);
void matTransposeMPI4(const std::vector<float>& mat, std::vector<float>& trans, int n, int mpi_rank, int mpi_size);

bool checkSymSerial(const vector<vector<float>>& M,int n);
int checkSymMPI(const std::vector<float>& mat, int n, int mpi_rank, int mpi_size);
bool checkSymOmp(const vector<vector<float>>& M,int n, int n_thread);

#endif