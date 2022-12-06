#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void calcolo_oracolo(float *a, float *b, float *oracolo, int n);

int main (void){
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    int M;
    float* h_a = 0;     // Host array a
    float* d_a;         // Device array a
    float* h_b = 0;     // Host array b
    float *d_b;         // Device array b
    float result = 0;   // Risultato finale
    float oracolo = 0.0;

    cudaEvent_t start, stop; // eventi per il calcolo del tempo di esecuzione
    float elapsed_time = 0.0;
    float tempo_sequenziale = 0.0;
	
    printf("Inserisci lunghezza vettori: ");
    scanf("%d", &M);

    h_a = (float *)malloc (M * sizeof (*h_a));      // Alloco h_a e lo inizializzo
    if (!h_a) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }
    
    h_b = (float *)malloc (M * sizeof (*h_b));  // Alloco h_b e lo inizializzo
    if (!h_b) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }

    srand(time(NULL));
    for (int i = 0; i < M; i++) {
        h_a[i] = ((float)rand() * 4 / (float)RAND_MAX) - 2; //reali nell'intervallo (-2,+2)
        h_b[i] = ((float)rand() * 4 / (float)RAND_MAX) - 2;
    }

    /* Calcolo Oracolo e tempo di esecuzione sequenziale */
    clock_t inizio = clock();    
    calcolo_oracolo(h_a, h_b, &oracolo, M);
    clock_t fine = clock();
    tempo_sequenziale = (float)(fine - inizio) / CLOCKS_PER_SEC;    

    cudaStat = cudaMalloc ((void**)&d_a, M*sizeof(*h_a));       // Alloco d_a
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        return EXIT_FAILURE;
    }
    
    cudaStat = cudaMalloc ((void**)&d_b, M*sizeof(*h_b));       // Alloco d_b
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        return EXIT_FAILURE;
    }
    
    stat = cublasCreate(&handle);               // Creo l'handle per cublas
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    
    stat = cublasSetVector(M,sizeof(float),h_a,1,d_a,1);    // Setto h_a su d_a
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree (d_a);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    
    stat = cublasSetVector(M,sizeof(float),h_b,1,d_b,1);    // Setto h_b su d_b
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree (d_b);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    /* Creiamo gli eventi per il tempo */
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); // tempo di inizio

    stat = cublasSdot(handle,M,d_a,1,d_b,1,&result);        // Calcolo il prodotto
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed cublasSdot");
        cudaFree (d_a);
        cudaFree (d_b);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    cudaEventRecord(stop); // tempo di fine

    /* Calcolo tempo di esecuzione */
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    printf("Risultato del prodotto --> %f\n",result);
    printf("Oracolo: %f\n", oracolo);
    printf("Tempo di esecuzione cublas: %fs\n", elapsed_time/1000);
    printf("Tempo di esecuzione sequenziale: %fs\n", tempo_sequenziale);
    printf("Speedup: %f\n", tempo_sequenziale/(elapsed_time/1000));

    /* Rilascio degli eventi */
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaFree (d_a);     // Dealloco d_a
    cudaFree (d_b);     // Dealloco d_b
    
    cublasDestroy(handle);  // Distruggo l'handle
    
    free(h_a);      // Dealloco h_a
    free(h_b);      // Dealloco h_b    
    return EXIT_SUCCESS;
}

void calcolo_oracolo(float *a, float *b, float *oracolo, int n) {
    for (int i = 0; i < n; i++) {
        *oracolo += a[i] * b[i];
    }
}
