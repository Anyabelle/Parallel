#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <omp.h>

void prt1a(char *t1, double *v, int n, char *t2);
void wtime(double *t)
{
	static int sec = -1;
	struct timeval tv;
	gettimeofday(&tv, (void *)0);
	if (sec < 0) sec = tv.tv_sec;

	*t = (tv.tv_sec - sec) + 1.0e-6 * tv.tv_usec;
}

int N;
double *A;
#define A(i,j) A[(i) * (N+1) + (j)]
double *X;

int main(int argc,char **argv){
	int NUM_THREADS = 2;
	if (atoi(argv[1])) {
		NUM_THREADS = atoi(argv[1]);
	}
	double time0, time1;
	wtime(&time0);
	int i, j, k;
	N = atoi(argv[2]);
	/* create arrays */
	A=(double *)malloc(N * (N + 1) * sizeof(double));
	X=(double *)malloc(N * sizeof(double));
	printf("GAUSS %dx%d\n----------------------------------\n",N,N);
	/* initialize array A*/
	for(i = 0; i <= N - 1; i++)
		for(j = 0; j <= N; j++) {
			A(i, j) = ((double)(rand() % 1000)) * 0.01;
		}
	/* elimination */
	int l, h;
	#pragma omp parallel for private (i, k, j, l, h) schedule(dynamic) num_threads(NUM_THREADS)
	for (i = 0; i < N; i++) {
		for (k = i + 1; k < N; k++) {
			double maxi = abs(A(i, i));
			int maxind = i; 
			for (j = i + 1; j < N; j++) {
				if (abs(A(j, i)) > maxi) {
					maxi = abs(A(j, i));
					maxind = j; 
				}
			}
			for (l = i; l < N; l++) {
				double tmp = A(i, l);
				A(i, l) = A(maxind, l);
				A(maxind, l) = tmp;
			}
			double c = A(k, i) / (A(i, i) + 1e-9);
			for (h = i; h <= N; h++) {
				A(k, h) = A(k, h) - c * A(i, h);
			}

		}
	}
	/* reverse substitution */
	X[N - 1] = A(N - 1, N) / (A(N - 1, N - 1) + 1e-9);
	#pragma omp parallel for private (i, j) schedule(dynamic) num_threads(NUM_THREADS)
	for (i = N - 2; i >= 0; i--) {
		for (j = i + 1; j < N; j++) {
			A(i, N) -= A(i, j) * X[j];
		}
		X[i] = A(i, N) / (A(i, i) + 1e-9);
	}
	wtime(&time1);
	printf("Time in seconds=%gs\n",time1-time0);
	prt1a("X = (", X, N > 9 ? 9 : N, "...)\n");
	free(A);
	free(X);
	return 0;
}

void prt1a(char *t1, double *v, int n, char *t2){
	int j;
	printf("%s", t1);
	for(j=0;j<n;j++)
		printf("%.4g%s", v[j], j % 10 == 9 ? "\n": ", ");
	printf("%s", t2);
}
