#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <random>
#include <omp.h>
#include <fstream> // For file I/O
#include "Functions.h"
#include <mpi.h>
#define TEST 5

using namespace std;
void writeToFile(const string& filename,const int dim_matrix, const double text, string num_threads_or_type_of_compile_opt = "0");

void initializeMatrix(vector<vector<float>>& matrix, int n) {     // Funzione per inizializzare una matrice n x n con numeri casuali a virgola mobile                                                                  
    random_device rd;                                             // Inizializzazione del generatore di numeri casuali
    mt19937 gen(rd());                                            
    uniform_real_distribution<> dis(0.0, 10.0);
    for (int i = 0; i < n; ++i) {                                  // Popolamento della matrice con valori casuali
        for (int j = 0; j < n; ++j) {
            float num = dis(gen);                                  // Numeri casuali tra 0 e 10
            matrix[i][j] = round(num* 100.0) / 100.0;
        }
    }
}
                                                                     // Funzione per stampare la matrice con allineamento perfetto
void printMatrix(const vector<vector<float>>& matrix, int n) {                                        // Numero di decimali (pu� essere cambiato)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {                                 // Stampa ogni numero con una larghezza fissa, precisione e spazio uniforme                           
            cout << matrix[i][j]<<"\t";
        }
        cout << endl;  
    }
}

bool checkTransposition(const vector<vector<float>>& M,int n,const vector<vector<float>>& T) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (M[j][i] != T[i][j]) {
                return false;
            }
        }
    }
    return true;
}

void writeToFile(const string& filename,const int dim_matrix, const double text, string num_threads_or_type_of_compile_opt) {
    ofstream file(filename,ios::app);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return;
    }
    if(num_threads_or_type_of_compile_opt != "0"){
        file <<dim_matrix <<";"<< text <<";"<< num_threads_or_type_of_compile_opt <<"\n";
    }else{
        file <<dim_matrix <<";"<< text <<"\n";
    }
    file.close();
    if (file.fail()) {
        cerr << "Error: Failed to close file " << filename << endl;
        return;
    }
}

vector<float> flatten(const vector<vector<float>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    vector<float> flatArray(rows * cols); // Creiamo un vettore unidimensionale
    int index = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            flatArray[index++] = matrix[i][j]; // Inseriamo ogni elemento in ordine nel vettore
        }
    }
    return flatArray; // Ritorniamo il vettore appiattito
}

vector<vector<float>> deflatten(const vector<float>& flatArray, int size) {
    // Creiamo una matrice quadrata di dimensione size x size
    vector<vector<float>> matrix(size, vector<float>(size));
    int index = 0;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i][j] = flatArray[index++];  // Ripristiniamo i valori nel formato bidimensionale
        }
    }
    return matrix;  // Ritorniamo la matrice ripristinata
}

void stampaFlat(const vector<float>& vec) {
    for (float val : vec) {
        cout << val << " ";
    }
    cout << endl;
}

int main(int argc, char* argv[]) {
    double wt1, wt2;  
    double Stime,Itime,Otime;                                           //for wall clock time
    int n_threads[8] = {1, 2, 4, 8, 16, 32, 64, 96};
    int MPI_sizes[9] = {4,5,6,7,8,9,10,11,12};
    
    MPI_Init (& argc , & argv );
    int rank, sizee;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &sizee);
    for (int size: MPI_sizes) {
        int n = pow(2, size);
        vector<vector<float>> M(n, vector<float>(n));                // Creiamo una matrice n x n
        vector<vector<float>> T(n, vector<float>(n));                //Matrice Trasposta
        initializeMatrix(M, n);                                      // Inizializziamo la matrice con valori casuali
        //printMatrix(M,n);
        //printMatrix(T,n);
        if (rank == 0){
          if(!checkSymSerial(M,n,1)){
            for (int i = 0; i < TEST; ++i) {
                    //Serial implementation---------------------------------------------
                    wt1 = omp_get_wtime();
                    matTransposeSerial(M,n,T,1);
                    wt2 = omp_get_wtime();
                    if(!checkTransposition(M,n,T)){cout<<"transpose not correct"<<endl;}
                    Stime += (wt2 - wt1);
                    writeToFile("../output/Serial.csv",size,(wt2 - wt1));     //--------------------------------------------write file Serial
            }
            cout <<"----------------------------------"<<endl;
            cout << "Serial Implemenation"<< endl;
            cout <<" "<<size<< "\t" << (Stime/TEST)<< " sec\t" << endl<<endl;
          }else{
            cout << "Symmetry corect"<< endl;
          }
  
          cout << "Omp Implemenation"<< endl;
          for (int& thread_count : n_threads) {
              if(!checkSymOmp(M,n,thread_count)){
                for (int i = 0; i < TEST; ++i) {
                    //Omp implementation-------------------------------------------------
                    wt1 = omp_get_wtime();
                    matTransposeOmp(M,n,T,thread_count);
                    wt2 = omp_get_wtime();
                    if(!checkTransposition(M,n,T)){cout<<"transpose not correct"<<endl;}
                    Otime += (wt2 - wt1);
                    writeToFile("../output/Omp.csv",size,(wt2 - wt1),to_string(thread_count)); //---------------------------write file OMP
                }
                cout <<" "<<size << "\t" << (Otime/TEST)<< " sec\t" <<thread_count<<" threads"<< endl;
    
                Otime = 0;
              }
          }
          
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0){cout <<endl<< "MPI Implemenation"<< endl;}
        if (n >= sizee || n%sizee==0){ // if matrix size >= number of processes
          //MPI implementation-------------------------------------------
          if (!checkSymMPI(M,n,1)){
            vector<float> M_f = flatten(M);
            vector<float> T_f(n*n,0);
            for (int i = 0; i < TEST; ++i) {
              MPI_Barrier(MPI_COMM_WORLD);
              wt1 = omp_get_wtime();
              matTransposeMPI(M_f ,T_f ,n ,rank ,sizee);
              MPI_Barrier(MPI_COMM_WORLD);
              wt2 = omp_get_wtime();
    
              if (rank == 0){
                  T = deflatten(T_f,n);
                  if(!checkTransposition(M,n,T)){cout<<"transpose not correct"<<endl;}
                  writeToFile("../output/MPI.csv",size,(wt2 - wt1),to_string(sizee));//-----------------------------------------write file implicit
    
              }
              Itime += (wt2 - wt1);
            }
            if (rank == 0){cout <<" "<<size<< "\t" << (Itime/TEST)<< " sec\t" <<sizee<<" Processes "<< endl;}
            Itime = 0;
          }
          if (rank == 0){
            // Stampa la matrice trasposta nel terminale
            //printMatrix(M, n);
            cout << endl;
            //printMatrix(T, n);  // Stampa la matrice trasposta            
          }
        }else{if (rank == 0){cout <<"To mutch processes for the number of rows or no processes at the power of 2  -----> ("<<n<<" rows for "<<sizee<<" Processes)"<< endl;}}
    }
    MPI_Finalize ();
    return 0;

}