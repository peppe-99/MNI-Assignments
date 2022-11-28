#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>

__global__ void prodottoScalare3(float *a, float *b, float *c, int n);
void prodottoScalareSequenziale(float *a, float *b, float *oracolo, int n);

int main(void) {

    float *a_host, *b_host, *c_host;
    float *a_dev, *b_dev, *c_dev;
    int n;
    float prodotto_scalare = 0.0, oracolo = 0.0;
    float elapsed_time = 0.0;

    cudaEvent_t start, stop;

    /* input: dimensione vettori */
    printf("Inserire dimensione vettori: ");
    scanf("%d", &n);

    dim3 blockDim(64);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x);
    printf("\nblockDim = %d", blockDim.x);
    printf("\ngridDim = %d", gridDim.x);

    /* Allocazione memoria host */
    a_host = (float*) malloc(n * sizeof(float));
    b_host = (float*) malloc(n * sizeof(float));
    c_host = (float*) malloc(gridDim.x * sizeof(float));

    /* Inizializzazione pseudocasuale dei vettori */
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        a_host[i] = ((float)rand() * 4 / (float)RAND_MAX) - 2; //reali nell'intervallo (-2,+2)
        b_host[i] = ((float)rand() * 4 / (float)RAND_MAX) - 2;
    }

    /* Calcolo dell'oracolo */
    prodottoScalareSequenziale(a_host, b_host, &oracolo, n);

    /* Allocazine memoria device */
    cudaMalloc((void **) &a_dev, n * sizeof(float));
    cudaMalloc((void **) &b_dev, n * sizeof(float));
    cudaMalloc((void **) &c_dev, gridDim.x * sizeof(float));

    /* Copia vettori da host a device */
    cudaMemcpy(a_dev, a_host, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b_host, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c_dev, c_host, gridDim.x * sizeof(float), cudaMemcpyHostToDevice);

    /* Creiamo gli eventi per il tempo */
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); // tempo di inizio

    /* invocazioen dinamica del kernel */
    prodottoScalare3<<<gridDim, blockDim, blockDim.x>>>(a_dev, b_dev, c_dev, n);

    /* copia dei risultati */
    cudaMemcpy(c_host, c_dev, gridDim.x * sizeof(float), cudaMemcpyDeviceToHost);

    /* somma sull'host */
    for (int i = 0; i < gridDim.x; i++) {
        prodotto_scalare += c_host[i];
    }

    cudaEventRecord(stop); // tempo di fine

    /* Calcolo tempo di esecuzione */
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

    /* stampa dei risultati */
    if (n <= 10) {
        printf("\nVettore a: ");
        for (int i = 0; i < n; i++) {
            printf("%f\t", a_host[i]);
        }
        printf("\nVettore b: ");
        for (int i = 0; i < n; i++) {
            printf("%f\t", b_host[i]);
        }
        printf("\nVettore c: ");
        for (int i = 0; i < gridDim.x; i++) {
            printf("%f\t", c_host[i]);
        }
    }
    printf("\nProdotto Scalare: %f\n", prodotto_scalare);
    printf("Oracolo: %f\n", oracolo);
    printf("Tempo di Esecuzione: %fs\n", elapsed_time/1000);

    /* Rilascio degli eventi */
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    /* free della memoria */
    free(a_host);
    free(b_host);
    free(c_host);
    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(c_dev);

    return 0;
}

__global__ void prodottoScalare3(float *a, float *b, float *c, int n) {
    /* vettore shared che conterrà i prodotti effettuati dai thread di un blocco */
    extern __shared__ float v[];

    /* ogni thread ricava gli indici su cui deve lavorare */
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int id = threadIdx.x;

    if (index < n) {
        /* prodotto componenti dei due vettori */
        v[id] = a[index] * b[index];

        /* sincronizzazione dei thread */
        __syncthreads();

        /* Somma parallela dei prodotti effettuati dai thread di un blocco */
        for (int dist = blockDim.x; dist > 1;) {
            dist = dist / 2;
            if (id < dist) {
                v[id] = v[id] + v[id + dist];
            }
            __syncthreads();
        }

        if (id == 0) {
            /* il thread 0 ha la somma finale dei prodotti effettuati dai thread di un blocco */
            c[blockIdx.x] = v[0];
        }
    }
}

void prodottoScalareSequenziale(float *a, float *b, float *oracolo, int n) {
    for (int i = 0; i < n; i++) {
        *oracolo += a[i] * b[i];
    }
}
