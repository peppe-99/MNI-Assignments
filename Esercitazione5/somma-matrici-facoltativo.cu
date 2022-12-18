#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <time.h>

__global__ void sommaMatrixGPU(double *a, double *b, double *s, int m);
void sommmaMatrixSequenziale(double *a, double *b, double *oracolo, int n, int m); 

int main(void) {
    double *a_host, *b_host, *s_host, *oracolo;
    double *a_dev, *b_dev, *s_dev;
    float elapsed_time = 0.0, tempo_sequenziale = 0.0;
    int n, m;

    cudaEvent_t start, stop;

    /* Input: dimensione della matrice */
    printf("Inserire dimensionioni delle matrici: ");
    scanf("%d %d", &n, &m);

    /* Allocazione memoria host */
    a_host = (double*) malloc((n * m) * sizeof(double));
    b_host = (double*) malloc((n * m) * sizeof(double));
    s_host = (double*) malloc((n * m) * sizeof(double));
    oracolo = (double*) malloc((n * m) * sizeof(double));

    /* Inizializzazione pseudocasuale delle matrici */
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            a_host[i * m + j] = ((double)rand() * 4 / (double)RAND_MAX) - 2;
            b_host[i * m + j] = ((double)rand() * 4 / (double)RAND_MAX) - 2;
        }
    }

    /* Calcolo Oracolo e Tempo d'esecuzione Sequenziale */
    clock_t inizio = clock();    
    sommmaMatrixSequenziale(a_host, b_host, oracolo, n, m);
    clock_t fine = clock();
    tempo_sequenziale = (float)(fine - inizio) / CLOCKS_PER_SEC; 

    /* Allocazione memoria device */
    cudaMalloc((void **) &a_dev, (n * m) * sizeof(double));
    cudaMalloc((void **) &b_dev, (n * m) * sizeof(double));
    cudaMalloc((void **) &s_dev, (n * m) * sizeof(double));

    /* Copia dei vettori da host a device */
    cudaMemcpy(a_dev, a_host, (n * m) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b_host, (n * m) * sizeof(double), cudaMemcpyHostToDevice);

    /* Configurazione del Kernel */
    dim3 blockDim(8, 8); // (8, 8) ottimale
    dim3 gridDim(
        (n + blockDim.x - 1) / blockDim.x,
        (m + blockDim.y - 1) / blockDim.y
    );
    printf("blockDim = (%d,%d)\n", blockDim.x, blockDim.y);
    printf("gridDim = (%d,%d)\n", gridDim.x, gridDim.y);

    /* Creiamo gli eventi per il tempo */
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); // tempo di inizio

    /* Inovcazione del Kernel */
    sommaMatrixGPU<<<gridDim, blockDim>>>(a_dev, b_dev, s_dev, m);

    /* Calcolo tempo di esecuzione */
    cudaEventRecord(stop); // tempo di fine
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

    /* Copia del risultato da device ad host */
    cudaMemcpy(s_host, s_dev, (n * m) * sizeof(double), cudaMemcpyDeviceToHost);

    /* Stampa dei risultati */
    if (n <= 5 && m <= 5) {
        printf("\nMatrice A:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                printf("%f\t", a_host[i * m + j]);
            }
            printf("\n");
        }
        printf("\nMatrice B:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                printf("%f\t", b_host[i * m + j]);
            }
            printf("\n");
        }
        printf("\nMatrice Somma:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                printf("%f\t", s_host[i * m + j]);
            }
            printf("\n");
        }
        printf("\nOracolo:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                printf("%f\t", s_host[i * m + j]);
            }
            printf("\n");
        }
    }
    printf("\nTempo di esecuzione parallelo (GPU): %fs\n", elapsed_time/1000);
    printf("\nTempo di esecuzione sequenziale (CPU): %fs\n", tempo_sequenziale);

    /* Rilascio degli eventi */
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    /* free della memoria */
    free(a_host);
    free(b_host);
    free(s_host);
    free(oracolo);
    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(s_dev);

    return 0;
}

__global__ void sommaMatrixGPU(double *a, double *b, double *s, int m) {
    int i = threadIdx.x + (blockDim.x * blockIdx.x);
    int j = threadIdx.y + (blockDim.y * blockIdx.y);
    s[i * m + j] = a[i * m + j] + b[i * m + j];
}

void sommmaMatrixSequenziale(double *a, double *b, double *oracolo, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            oracolo[i * m + j] = a[i * m + j] + b[i * m + j];
        }
    }
} 


