#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void calcolo_oracolo(float *a, float *b, float *oracolo, int rows, int cols);

int main (void){
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    int rows, cols;
    float* h_a = 0;     // Host matrix a
    float* d_a;         // Device matrix a
    float* h_b = 0;     // Host array b
    float *d_b;         // Device array b
    float *h_result = 0;   // Risultato finale
    float *d_result = 0;   // Risultato finale
    float *oracolo = 0;

    cudaEvent_t start, stop; // eventi per il calcolo del tempo di esecuzione
    float elapsed_time = 0.0;
    float tempo_sequenziale = 0.0;
	
    printf("Inserisci dimensioni matrice: ");
    scanf("%d %d", &rows, &cols);

    h_a = (float *)malloc (rows * cols * sizeof (*h_a));  // Alloco h_a
    if (!h_a) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }
    
    h_b = (float *)malloc (cols * sizeof (*h_b));  // Alloco h_b
    if (!h_b) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }

    h_result = (float *)malloc (rows * sizeof(*h_result));  // Alloco result
    if (!h_result) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }

    oracolo = (float *)malloc (rows * sizeof(*oracolo));    // Alloco l'oracolo
    if (!oracolo) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }

    /* Inizializzazione pseudocasuale della matrice e del vettore */
    srand(time(NULL));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            h_a[i * cols + j] = ((float)rand() * 4 / (float)RAND_MAX) - 2; //reali nell'intervallo (-2,+2)
        } 
    }

    for (int i = 0; i < cols; i++) {
        h_b[i] = ((float)rand() * 4 / (float)RAND_MAX) - 2;
    }

    /* Calcolo Oracolo e tempo di esecuzione sequenziale */
    clock_t inizio = clock();    
    calcolo_oracolo(h_a, h_b, oracolo, rows, cols);
    clock_t fine = clock();
    tempo_sequenziale = (float)(fine - inizio) / CLOCKS_PER_SEC;

    cudaStat = cudaMalloc ((void**)&d_a, rows * cols * sizeof(*h_a));   // Alloco d_a
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        return EXIT_FAILURE;
    }
    
    cudaStat = cudaMalloc ((void**)&d_b, cols * sizeof(*h_b));  // Alloco d_b
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        return EXIT_FAILURE;
    }

    cudaStat = cudaMalloc ((void**)&d_result, rows * sizeof(*h_result));  // Alloco d_result
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        return EXIT_FAILURE;
    }
    
    stat = cublasCreate(&handle);   // Creo l'handle per cublas
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    stat = cublasSetMatrix(rows, cols, sizeof(float), h_a, rows, d_a, rows); // Setto h_a su d_a
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed matrix");
        cudaFree (d_a);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    
    stat = cublasSetVector(cols, sizeof(float),h_b,1,d_b,1);    // Setto h_b su d_b
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed vector");
        cudaFree (d_b);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    /* Creiamo gli eventi per il tempo */
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); // tempo di inizio

    float alpha = 1.0;
    float beta = 1.0;
    stat = cublasSgemv(handle, CUBLAS_OP_T, rows, cols, &alpha, d_a, rows, d_b, 1, &beta, d_result, 1); // calcolo prodotto matrice-vettore
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed cublasSdot");
        cudaFree (d_a);
        cudaFree (d_b);
        cudaFree (d_result);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    cudaEventRecord(stop); // tempo di fine

    /* Calcolo tempo di esecuzione */
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

    stat = cublasGetVector(rows, sizeof(float), d_result, 1, h_result, 1); // ottengo il risultato

    if (rows <= 10 && cols <= 10) {
        printf("\nMatrice:\n");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                printf("%f\t", h_a[i * cols + j]);
            }
            printf("\n");
        }

        printf("\nVettore:\n");
        for (int i = 0; i < cols; i++) {
            printf("%f\n", h_b[i]);
        }
            
        printf("\nVettore Prodotto:\n");
        for (int i = 0; i < rows; i++) {
            printf("%f\n", h_result[i]);
        }

        printf("\nOracolo:\n");
        for (int i = 0; i < rows; i++) {
            printf("%f\n", oracolo[i]);
        }
    }
    printf("\nTempo di esecuzione cublas: %fs\n", elapsed_time/1000);
    printf("Tempo di esecuzione sequenziale: %fs\n", tempo_sequenziale);
    printf("Speedup: %f\n", tempo_sequenziale/(elapsed_time/1000));

    /* Rilascio degli eventi */
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaFree (d_a);         // Dealloco d_a
    cudaFree (d_b);         // Dealloco d_b
    cudaFree (d_result);    // Dealloco d_result
    
    cublasDestroy(handle);  // Distruggo l'handle
    
    free(h_a);      // Dealloco h_a
    free(h_b);      // Dealloco h_b
    free(h_result); // Dealloco h_result
    free(oracolo);  // Dealloco oracolo

    return EXIT_SUCCESS;
}

void calcolo_oracolo(float *a, float *b, float *oracolo, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        oracolo[i] = 0;
        for (int j = 0; j < cols; j++) {
            oracolo[i] += a[i * cols + j] * b[j];
        }
    }
}
