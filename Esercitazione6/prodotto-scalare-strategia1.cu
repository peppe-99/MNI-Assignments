#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <time.h>

__global__ void prodottoGPU(double *u, double *v, double *w, int n);
void prodottoScalareSequenziale(double *u, double *v, double *oracolo, int n);

int main(void) {
    double *u_host, *v_host, *w_host;
    double *u_dev, *v_dev, *w_dev;
    double prod_scalare = 0.0, oracolo = 0.0;
    float elapsed_time, tempo_sequenziale;
    int N;

    cudaEvent_t start, stop;

    /* Input: lunghezza dei vettori */
    printf("Inserisci numero elementi dei vettori: ");
    scanf("%d", &N);

    /* Calcolo dimensioni della griglia */
    dim3 blockDim(64);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
    printf("\nblockDim: %d\n", blockDim.x);
    printf("gridDim: %d\n", gridDim.x);

    /* Allocazione memoria host */
    u_host = (double*)malloc(N * sizeof(double));
    v_host = (double*)malloc(N * sizeof(double));
    w_host = (double*)malloc(N * sizeof(double));

    /* Inizializzazione pseucodasuale dei vettori */
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        u_host[i] = ((double)rand() * 4 / (double)RAND_MAX) - 2; //reali nell'intervallo (-2,+2)
        v_host[i] = ((double)rand() * 4 / (double)RAND_MAX) - 2;
    }

    /* Calcolo dell'oracolo e del tempo di esecuzione sequenziale */
    clock_t inizio = clock();    
    prodottoScalareSequenziale(u_host, v_host, &oracolo, N);
    clock_t fine = clock();
    tempo_sequenziale = (float)(fine - inizio) / CLOCKS_PER_SEC;    

    /* Allocazione memoria device */
    cudaMalloc((void **) &u_dev, N * sizeof(double));
    cudaMalloc((void **) &v_dev, N * sizeof(double));
    cudaMalloc((void **) &w_dev, N * sizeof(double));

    /* Copia dei vettori da host a device */
    cudaMemcpy(u_dev, u_host, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(v_dev, v_host, N * sizeof(double), cudaMemcpyHostToDevice);

    /* Creiamo gli eventi per il tempo */
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); // tempo di inizio

    /* Invocazione del Kernel */
    prodottoGPU<<<gridDim, blockDim>>>(u_dev, v_dev, w_dev, N);

    /* Copa del risultato da device a host */
    cudaMemcpy(w_host, w_dev, N * sizeof(double), cudaMemcpyDeviceToHost);

    /* Somma seriale su CPU */
    for (int i = 0; i < N; i++) {
        prod_scalare += w_host[i];
    }
        
    cudaEventRecord(stop); // tempo di fine
    
    /* Calcolo tempo di esecuzione*/
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

    /* Stampa dei risultati */
    if (N <= 10) {
        printf("\nVettore v: ");
        for (int i = 0; i < N; i++) {
            printf("%f\t", v_host[i]);
        }
        printf("\nVettore u: ");
        for (int i = 0; i < N; i++) {
            printf("%f\t", u_host[i]);
        }
        printf("\nVettore w: ");
        for (int i = 0; i < N; i++) {
            printf("%f\t", w_host[i]);
        }
    }
    printf("\n\nOracolo: %f\n", oracolo);
    printf("Prodotto Scalare: %f\n", prod_scalare);
    printf("\nTempo di esecuzione parallelo (GPU): %fs\n", elapsed_time/1000);
    printf("\nTempo di esecuzione sequenziale (CPU): %fs\n", tempo_sequenziale);

    /* Rilascio degli eventi */
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    /* free della memoria host e device */
    free(u_host);
    free(v_host);
    free(w_host);
    cudaFree(u_dev);
    cudaFree(v_dev);
    cudaFree(w_dev);

    return 0;
}

__global__ void prodottoGPU(double *u, double *v, double *w, int n) {
    /* Ogni thread ricava su quali elementi deve lavorare */
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        w[index] = v[index] * u[index];
    }
}

void prodottoScalareSequenziale(double *u, double *v, double *oracolo, int n) {
    for (int i = 0; i < n; i++) {
        *oracolo += u[i] * v[i];
    }
}

