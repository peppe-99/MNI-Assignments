#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <time.h>

__global__ void sommaMatrixGPU(double *a, double *b, double *s, int n);

int main(void) {
    double *a_host, *b_host, *s_host;
    double *a_dev, *b_dev, *s_dev;
    float elapsed_time;
    int n;

    cudaEvent_t start, stop;

    /* Input: dimensione della matrice */
    printf("Inserire dimensione della matrice: ");
    scanf("%d", &n);

    /* Allocazione memoria host */
    a_host = (double*) malloc((n * n) * sizeof(double));
    b_host = (double*) malloc((n * n) * sizeof(double));
    s_host = (double*) malloc((n * n) * sizeof(double));

    /* Inizializzazione pseudocasuale delle matrici */
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a_host[i * n + j] = ((double)rand() * 4 / (double)RAND_MAX) - 2;
            b_host[i * n + j] = ((double)rand() * 4 / (double)RAND_MAX) - 2;
        }
    }

    /* Allocazione memoria device */
    cudaMalloc((void **) &a_dev, (n * n) * sizeof(double));
    cudaMalloc((void **) &b_dev, (n * n) * sizeof(double));
    cudaMalloc((void **) &s_dev, (n * n) * sizeof(double));

    /* Copia dei vettori da host a device */
    cudaMemcpy(a_dev, a_host, (n * n) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b_host, (n * n) * sizeof(double), cudaMemcpyHostToDevice);

    /* Configurazione del Kernel */
    dim3 blockDim(32, 32);
    dim3 gridDim(
        (n + blockDim.x - 1) / blockDim.x,
        (n + blockDim.y - 1) / blockDim.y
    );

    /* Creiamo gli eventi per il tempo */
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); // tempo di inizio

    /* Inovcazione del Kernel */
    sommaMatrixGPU<<<blockDim, gridDim>>>(a_dev, b_dev, s_dev, n);

    /* Copia del risultato da device ad host */
    cudaMemcpy(s_host, s_dev, (n * n) * sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop); // tempo di fine
    cudaEventSynchronize(stop);

    /* Stampa dei risultati */
    cudaEventElapsedTime(&elapsed_time, start, stop);
    if (n <= 5) {
        printf("\nMatrice A:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%f\t", a_host[i * n + j]);
            }
            printf("\n");
        }
        printf("\nMatrice B:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%f\t", b_host[i * n + j]);
            }
            printf("\n");
        }
        printf("\nMatrice Somma:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%f\t", s_host[i * n + j]);
            }
            printf("\n");
        }
    }
    printf("\n\nTempo di esecuzione: %f\n", elapsed_time);

    /* free della memoria */
    free(a_host);
    free(b_host);
    free(s_host);
    cudaFree(a_host);
    cudaFree(b_host);
    cudaFree(s_host);

    return 0;
}

__global__ void sommaMatrixGPU(double *a, double *b, double *s, int n) {
    int i = threadIdx.x + (blockDim.x * blockIdx.x);
    int j = threadIdx.y + (blockDim.y * blockIdx.y);
    s[i * n + j] = a[i * n + j] + b[i * n + j];
}

