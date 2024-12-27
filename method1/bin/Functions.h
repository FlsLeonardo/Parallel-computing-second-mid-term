#ifndef FUNCTIONS_H
#define FUNCTIONS_H
#include <vector>
using namespace std;

extern void (*matTranspose)(vector<vector<float>>& M,int n,vector<vector<float>>& T,int n_thread);
extern bool (*checkSym)(const vector<vector<float>>& M,int n, int n_thread);

#endif