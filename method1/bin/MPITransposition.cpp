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