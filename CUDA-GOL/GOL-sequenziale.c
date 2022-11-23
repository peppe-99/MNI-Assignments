#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#define SEED 28

/* ASCII codes */
#define LIVE "\xF0\x9F\x91\xBE"
#define DEATH "\xF0\x9F\x94\xB2"

/* Funzione per lo swap delle matrici di lettura e scrittura */
void swap(int **current_matrix, int **new_matrix);
/* Funzione per il calcolo dei vicini vivi */
int neighbors_alive(int i, int j, int rows, int cols, int *matrix);

int main(void) {
    int row, col;
    int rounds, cell_alive;
    int *read_matrix, *write_matrix;
    float tempo;

    /* Input: dimensioni matrice e numero di round */
    printf("Inserire dimensioni griglia: ");
    scanf("%d %d", &row, &col);
    printf("Inserire numero di round: ");
    scanf("%d", &rounds);

    /* Allocazione delle strutture dati */
    read_matrix = (int*)malloc(row * col * sizeof(int));
    write_matrix = (int*)malloc(row * col * sizeof(int));

    /* Inizializzazion pseudocasuale della matrice */
    srand(SEED);
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            read_matrix[i * col + j] = rand() % 2; // 0 oppure 1
        }
    }

    /* Stampa generazione iniziale */
    if (row <= 10 && col <= 10) {
        printf("\nGenerazione Iniziale:\n");
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                (read_matrix[i * col + j] == 1) ? printf(LIVE) : printf(DEATH);
            }
            printf("\n");
        }
    }

    /* Calcolo tempo di inizio */
    clock_t start = clock();
    
    /* Calcolo delle generazioni */
    for (int round = 0; round < rounds; round++) {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {

                /* Calcolo celle vicine vive */
                cell_alive = neighbors_alive(i, j, row, col, read_matrix);

                /* Aggiornamento della matrice */
                if (read_matrix[i * col + j] == 1 && (cell_alive == 2 || cell_alive == 3)) {
                    write_matrix[i * col + j] = 1;
                }
                else if (read_matrix[i * col + j] == 0 && cell_alive == 3) {
                    write_matrix[i * col + j] = 1;
                }
                else write_matrix[i * col + j] = 0;
            }
        }
        /* Swap matrici lettura e scrittura */
        swap(&read_matrix, &write_matrix);
    }

    /* Calcolo tempo di fine */
    clock_t end = clock();

    tempo = (float)(end - start) / CLOCKS_PER_SEC;    

    /* Stampa risultati */
    if (row <= 10 && col <= 10) {
        printf("\nGenerazione %d:\n", rounds);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                (read_matrix[i * col + j] == 1) ? printf(LIVE) : printf(DEATH);
            }
            printf("\n");
        }
    }
    printf("\nTempo di esecuzione sequenziale: %fs\n", tempo);

    return 0;
}

void swap(int **current_matrix, int **new_matrix) {
    int *temp = *current_matrix;
    *current_matrix = *new_matrix;
    *new_matrix = temp;
}

int neighbors_alive(int i, int j, int rows, int cols, int *matrix) {
    /* vicini superiori */
    int top = 0, tl_corner = 0, tr_corner = 0;
    if (i > 0) {
        top = matrix[(i-1) * cols + j];        
        if (j > 0) {
            tl_corner = matrix[(i-1) * cols + (j-1)];
        }
        if (j < cols-1) {
            tr_corner = matrix[(i-1) * cols + (j+1)];
        }
    }

    /* vicini laterali */
    int left = (j > 0) ? matrix[i * cols + (j-1)] : 0;
    int right = (j < cols-1) ? matrix[i * cols + (j+1)] : 0;

    /* vicini inferiori */
    int bottom = 0, bl_corner = 0, br_corner = 0;
    if (i < rows-1) {
        bottom = matrix[(i+1) * cols + j];
        if (j > 0) {
            bl_corner = matrix[(i+1) * cols + (j-1)];
        }
        if (j < cols-1) {
            br_corner = matrix[(i+1) * cols + (j+1)];
        }
    }

    return top + left + right + bottom + tl_corner + tr_corner + bl_corner + br_corner;
}