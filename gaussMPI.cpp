#include <iostream>
#include <cmath>
#include <mpi.h>
#include <fstream>
#include <ctime>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
using namespace std;
void swap(double** la, int cnt, int row1, int row2);
void giveProcess(vector<double> a, double* b, int cnt);
int rank, size;
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(argc < 2) {
        MPI_Finalize();
        return 0;
    } 
    double time0, time1, diff;
    int ctrlcurrent = 0;
    double *send;
    double *receive;
    double **data;
    vector<double> input;
    int cntrow, cntcol;
    if(rank == 0) {
        cntrow = atoi(argv[1]);
        input.resize(cntrow + 1);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double *X = new double[cntrow];  
    send = new double[cntrow + 1];
    MPI_Bcast (&cntrow, 1, MPI_INT, 0, MPI_COMM_WORLD);
    cntcol = (cntrow + 1) / size; 
    data = new double*[cntcol];
    double **data3 = new double *[cntrow];
    double *data2 = new double[cntrow * (cntrow + 1)];
    for(int i = 0; i < cntcol; i++) {
        data[i] = new double[cntrow];
    }
    for(int i = 0; i < cntcol; i++)
    {
        for(int j = 0; j < cntrow; j++)
            data[i][j] = 0;
    }
    for (int i = 0; i < cntrow; i++)
        data3[i]= new double[cntrow + 1];
    receive = new double[cntcol];
    for(int i = 0; i < cntrow; i++)
    {
        if(rank == 0) {
            for(int j = 0; j < cntrow + 1; j++) {
                input[j] = ((double)(rand() % 1000)) * 0.01;
                giveProcess(input, send, cntrow + 1);
            }
        }
        MPI_Scatter(send, cntcol, MPI_DOUBLE, receive, cntcol, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        for(int j = 0; j < cntcol; j++) {
            data[j][i] = receive[j];
        }
    }
    delete [] send;
    delete [] receive;
    MPI_Barrier(MPI_COMM_WORLD);
    time0 = MPI_Wtime();    
    send = new double[cntrow + 1];
    int current = 0;
    int swaps = 0;
    int cur_index = 0;
    for(int i = 0; i < cntrow; i++) { 
        int rowSwap;
        if(ctrlcurrent == rank) {
            rowSwap = current;
            double max = data[cur_index][current];
            for(int j = current + 1; j < cntrow; j++) {
                if(data[cur_index][j] > max) {
                    rowSwap = j;
                    max = data[cur_index][j];
                }
            }
        }
        MPI_Bcast(&rowSwap, 1, MPI_INT, ctrlcurrent, MPI_COMM_WORLD);
        if(rowSwap != current) {
            swap(data, cntcol, current, rowSwap);
            swaps++;
        }
        
        if(ctrlcurrent == rank) {
            for(int j = current; j < cntrow; j++)
                send[j] = data[cur_index][j] / data[cur_index][current];
        }
        MPI_Bcast(send, cntrow, MPI_DOUBLE, ctrlcurrent, MPI_COMM_WORLD);
        for(int j = 0; j < cntcol; j++)
        {
            for(int k = current + 1; k < cntrow; k++) {
                data[j][k] -= send[k] * data[j][current] ;
            }
        }
        if(ctrlcurrent == rank)
        {
            cur_index++;
        }
        ctrlcurrent++;
        if(ctrlcurrent == size)
            ctrlcurrent = 0;
        current++;
    }    
    MPI_Barrier(MPI_COMM_WORLD);
    double *datas = new double[cntrow * (cntrow + 1)];
    double *dat = new double[cntcol * cntrow];
    int cnt = 0;
    for(int i = 0; i < cntcol; i++)
    {
        for(int j = 0; j < cntrow; j++) {
            dat[cnt] = data[i][j];
            cnt++;
        }
    }
    MPI_Gather(dat, cntrow * cntcol,
              MPI_DOUBLE, datas, cntrow * cntcol,
              MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    time1 = MPI_Wtime();
    diff = time1 - time0;

    MPI_Barrier(MPI_COMM_WORLD);
    if(!rank)
    { 
        int now = 0;
        int c = 0;
        int procs = size;
        int cntrows = (cntrow + 1) / procs;
        int prnum = 0;
        for (int i = 0; i < cntrow; i++) {
            for (int j = 0; j < cntrow; j++) {
                data3[prnum + size * c][j] = datas[now];
                now++;
            }
            c++;
            if (c == cntrows) {
              c = 0;
              prnum++;
            }
        }
        for (int i = 0; i < cntrow; i++) {
            data3[i][cntrow] = datas[now];
            now++;
        }
        X[cntrow - 1] = data3[cntrow - 1][cntrow] / (data3[cntrow - 1][cntrow - 1] + 1e-7);
        for (int i = cntrow - 2; i >= 0; i--) {
          for (int j = i + 1; j < cntrow; j++) {
            data3[i][cntrow] -= data3[i][j] * X[j];
          }
          X[i] = data3[i][cntrow] / (data3[i][i] + 1e-7);
        }
        cout << "Run Time: " << diff  * 3.0<< endl;
        for (int i = 0; i < 10; i++) {
            cout << X[i] << " ";
        }
        cout << "\n";
    }
    delete [] send;
    for(int i = 0; i < cntcol; i++)
        delete [] data[i];
    for(int i = 0; i < cntrow; i++)
        delete [] data3[i];
    delete [] data3;
    delete [] data;
    delete [] data2;
    MPI_Finalize();
    return 0;
} 

void swap(double** la, int cnt, int row1, int row2)
{
    double tmp;
    if(row1 == row2)
        return;
    for(int i = 0; i < cnt; i++) {
        tmp = la[i][row1];
        la[i][row1] = la[i][row2];
        la[i][row2] = tmp;
    }
    return;
}
 
void giveProcess(vector<double> b, double* a, int cnt)
{
    int index = 0;
    for(int i = 0; i < size; i++) {
        for(int j = i; j < cnt; j += size) {
            a[index] = b[j];
            index++;
        }
    }
    return;
}


