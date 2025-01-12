#ifndef FUNCTIONS_H
#define FUNCTIONS_H
#include <vector>
using namespace std;

void matTransposeSerial(vector<vector<float>>& M,int n,vector<vector<float>>& T);      
void matTransposeOmp(vector<vector<float>>& M,int n,vector<vector<float>>& T, int n_thread);
void matTransposeMPI(const std::vector<float>& mat, std::vector<float>& local_trans, int n, int mpi_rank, int start_row ,int end_row);


bool checkSymSerial(const vector<vector<float>>& M,int n);
int checkSymMPI(const std::vector<float>& mat, int n, int mpi_rank, int mpi_size);
bool checkSymOmp(const vector<vector<float>>& M,int n, int n_thread);

#endif