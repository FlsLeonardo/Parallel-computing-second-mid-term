#ifndef FUNCTIONS_H
#define FUNCTIONS_H
#include <vector>
using namespace std;

void matTransposeSerial(vector<vector<float>>& M,int n,vector<vector<float>>& T,int n_thread);      
void matTransposeMPI(vector<vector<float>>& M,int n,vector<vector<float>>& T,int n_thread);
void matTransposeOmp(vector<vector<float>>& M,int n,vector<vector<float>>& T, int n_thread);
void matTransposeMPI(const vector<float>& flatMatrix, vector<float>& transposedMatrix, int matSize, int rank, int numProc);

bool checkSymSerial(const vector<vector<float>>& M,int n, int n_thread);
bool checkSymMPI(const vector<vector<float>>& M,int n, int n_thread);
bool checkSymOmp(const vector<vector<float>>& M,int n, int n_thread);

#endif