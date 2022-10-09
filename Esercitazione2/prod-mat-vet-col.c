/**
 *  Algoritmo parallelo per il prodotto matrice-vettore
 *  strategia: blocchi di colonne
*/

#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>

int main(int argc, char *argv[]) {
    
    int np, rank;
    int row, col, local_col;
    int *matrix, *vet, *vet_prod;
    int *local_matrix, *local_vet, *local_vet_prod;

    /* Inizializzazione dell'ambiente MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    if (rank == 0) {
        /* Ottenimento del numero di righe e colonne*/
        printf("Inserisci dimensione matrice: ");
        fflush(stdout);
        scanf("%d %d", &row, &col);

        /* Allocazione della matrice e del vettore */
        matrix = (int*)malloc(row * col * sizeof(int));
        vet = (int*)malloc(row * sizeof(int));
        vet_prod = (int*)malloc(row * sizeof(int));

        /* Generazione pseudocasuale di matrice e vettore */
        srand(28);
        printf("\nMatrice:\n");
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                matrix[i*col+j] = rand() % 10 + 1;
                printf("%d\t", matrix[i*col+j]);
            }   
            vet[i] = rand() % 10 + 1;
            printf("\n");
        }

        printf("\nVettore:\n");
        for (int i = 0; i < row; i++) printf("%d\n", vet[i]);

        /* Calcolo numero di colonne per ogni processore */
        local_col = col / np;
    }
    /* Invio in broadcast del numero di righe, colonne e colonne locali */
    MPI_Bcast(&row, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&col, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&local_col, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Allocazione strutture locali ad ogni processore */
    local_matrix = (int*)malloc(row * local_col * sizeof(int));
    local_vet = (int*)malloc(local_col * sizeof(int));
    local_vet_prod = (int*)malloc(row * sizeof(int));

    /* Suddivisione per colonne delle matrice */
    for (int i = 0; i < row; i++) {
        MPI_Scatter(&matrix[i*col], local_col, MPI_INT, &local_matrix[i*local_col], local_col, MPI_INT, 0, MPI_COMM_WORLD);
    }
    /* Suddivisione del vettore per righe */
    MPI_Scatter(vet, local_col, MPI_INT, local_vet, local_col, MPI_INT, 0, MPI_COMM_WORLD);


    /*printf("\nLOCAL MATRIX %d\n", rank);
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < local_col; j++) {
            printf("%d\t", local_matrix[i*local_col+j]);
        }
        printf("\n");
    }
    printf("\nLOCAL VET %d\n", rank);
    for (int i = 0; i < local_col; i++) {
        printf("%d\n", local_vet[i]);
    }*/


    /* Prodotto matrice-vettore locale */
    for (int i = 0; i < row; i++) {
        local_vet_prod[i] = 0;
        for (int j = 0; j < local_col; j++) {
            local_vet_prod[i] += local_matrix[i*local_col+j] * local_vet[j];
        }
    }

    /*printf("\nLOCAL VET PROD %d\n", rank);
    for (int i = 0; i < row; i++) {
        printf("%d\n", local_vet_prod[i]);
    }*/
    
    /* Riduzione con somma dei vettori prodotto locali */
    MPI_Reduce(local_vet_prod, vet_prod, row, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank==0) {
        fflush(stdout);
        printf("\nVettore Prodotto:\n");
        for (int i = 0; i<row; i++) {
            printf("%d\n", vet_prod[i]);
        }
    }


    MPI_Finalize();

    if (rank == 0) {
        free(matrix);
        free(vet);
        free(vet_prod);
    }
    free(local_vet);
    free(local_matrix);
    free(local_vet_prod);

    return 0;
}
