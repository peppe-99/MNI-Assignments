#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<cuda.h>

#define SEED 28

/* Codici ASCII */
#define LIVE "\xF0\x9F\x91\xBE"
#define DEATH "\xF0\x9F\x94\xB2"

/* Funzione invocata dall'host per scambiare le matrici */
void swap(int **current_matrix, int **new_matrix);

/* Kernel per calcolare la generazione successiva data l'attuale */
__global__ void gameOfLife(int *read_matrix, int *write_matrix, int row, int col);

/* Funzione invocata dal device per calcolare i vicini vivi di una cella */
__device__ int countCellAlive(int i, int j, int row, int col, int *matrix);

int main (void) {

    /* Variabili utilizzate */
    int row, col, generations;
    int *matrix_host;
    int *read_matrix_dev, *write_matrix_dev;
    float elapsed_time = 0.0;

    /* Eventi per il calcolo del tempo di esecuzione */
    cudaEvent_t start, stop;

    /* Input: dimensioni della matrice e numero di generazioni */
    printf("Inserisci dimensioni della matrice: ");
    scanf("%d %d", &row, &col);
    printf("Inserisci numero di generazioni: ");
    scanf("%d", &generations);

    /* Allocazione memoria host */
    matrix_host = (int*) malloc(row * col * sizeof(int));

    /* Inizializzazione pseudocasuale delle matrice (generazione iniziale) */
	//srand(SEED);
    srand(time(NULL));
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            matrix_host[i * col + j] = rand() % 2; // 0 oppure 1
        }
    }

    /* Stampa generazione iniziale */
    if (row <= 10 && col <= 10) {
        printf("\nGenerazione Iniziale:\n");
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                (matrix_host[i * col + j] == 1) ? printf(LIVE) : printf(DEATH);
            }
            printf("\n");
        }
    }

    /* Allocazione memoria device */
    cudaMalloc((void **) &read_matrix_dev, (row * col) * sizeof(int));
    cudaMalloc((void **) &write_matrix_dev, (row * col) * sizeof(int));

    /* Copia della matrice da host a device */
    cudaMemcpy(read_matrix_dev, matrix_host, (row * col) * sizeof(int), cudaMemcpyHostToDevice);

    /* Configurazione del Kernel */
    dim3 blockDim(8,12);
    dim3 gridDim(
        (col + blockDim.x - 1) / blockDim.x,
        (row + blockDim.y - 1) / blockDim.y
    );
    printf("\nblockDim = (%d,%d)\n", blockDim.x, blockDim.y);
    printf("gridDim = (%d,%d)\n", gridDim.x, gridDim.y);

    /* Creiamo gli eventi per il tempo */
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); // tempo di inizio

    /* Simulazione delle generazioni */
    for (int gen = 0; gen < generations; gen++) {
        /* Invocazione del Kernel */
        gameOfLife<<<gridDim, blockDim>>>(read_matrix_dev, write_matrix_dev, row, col);
    
        /* swap delle matrici */
        swap(&read_matrix_dev, &write_matrix_dev);
    }

    /* Calcolo tempo di esecuzione */
    cudaEventRecord(stop); // tempo di fine
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

    /* Copia della matrice con l'ultima generazioen da device ad host */
    cudaMemcpy(matrix_host, read_matrix_dev, (row * col) * sizeof(int), cudaMemcpyDeviceToHost);

    /* Stampa dei risultati */
    if (row <= 10 && col <= 10) {
        printf("\nGenerazione %d:\n", generations);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                (matrix_host[i * col + j] == 1) ? printf(LIVE) : printf(DEATH);
            }
            printf("\n");
        }
    }
    printf("\nTempo di esecuzione parallelo (GPU): %fs\n", elapsed_time/1000);

    /* free della memoria */
    free(matrix_host);
    cudaFree(read_matrix_dev);
    cudaFree(write_matrix_dev);

    return 0;
}

__global__ void gameOfLife(int *read_matrix, int *write_matrix, int row, int col) {
    /* Calcoliamo gli indici su cui dovrà lavorare un thread */
    int i = threadIdx.x + (blockDim.x * blockIdx.x);
    int j = threadIdx.y + (blockDim.y * blockIdx.y);

    if (i < col && j < row) {
        /* Calcolo del numero di celle vicine vive */
        int alive = countCellAlive(i, j, row, col, read_matrix);
            
        /* Applicazione regole del gioco */
        if (read_matrix[i * col + j] == 1 && (alive == 2 || alive == 3)) {
            write_matrix[i * col + j] = 1;
        }
        else if (read_matrix[i * col + j] == 0 && alive == 3) {
            write_matrix[i * col + j] = 1;
        }
        else write_matrix[i * col + j] = 0;
    }
}

__device__ int countCellAlive(int i, int j, int row, int col, int *matrix) {
    int alive = 0;

    /* vicini superiori */
    if (i > 0) {
        alive += matrix[(i-1) * col + j];        
        if (j > 0) {
            alive += matrix[(i-1) * col + (j-1)];
        }
        if (j < col-1) {
            alive += matrix[(i-1) * col + (j+1)];
        }
    }

    /* vicini laterali */
    alive += (j > 0) ? matrix[i * col + (j-1)] : 0;
    alive += (j < col-1) ? matrix[i * col + (j+1)] : 0;

    /* vicini inferiori */
    if (i < row-1) {
        alive += matrix[(i+1) * col + j];
        if (j > 0) {
            alive += matrix[(i+1) * col + (j-1)];
        }
        if (j < col-1) {
            alive += matrix[(i+1) * col + (j+1)];
        }
    }
    return alive;
}

void swap(int **current_matrix, int **new_matrix) {
    int *temp = *current_matrix;
    *current_matrix = *new_matrix;
    *new_matrix = temp;
}
