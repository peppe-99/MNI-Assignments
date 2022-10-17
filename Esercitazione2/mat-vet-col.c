/**
 *  Algoritmo parallelo per il prodotto matrice-vettore
 *  strategia: blocchi di colonne
*/

#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<mpi.h>

int main(int argc, char *argv[]) {
    
    int np, rank;
    int row, col, local_col;
    double *matrix, *vet, *vet_prod;
    double *local_matrix, *local_vet, *local_vet_prod;

    double T_inizio, T_fine, T_max;


    /* Inizializzazione dell'ambiente MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    if (rank == 0) {
        /* Ottenimento del numero di righe e colonne*/
        printf("Inserisci numero di righe: ");
        fflush(stdout);
        scanf("%d", &row);
        
        printf("Inserisci numero di colonne: ");
        fflush(stdout);
        scanf("%d", &col);

        /* Allocazione della matrice e del vettore */
        matrix = (double*)malloc(row * col * sizeof(double));
        vet = (double*)malloc(col * sizeof(double));
        vet_prod = (double*)malloc(row * sizeof(double));

        /* Generazione pseudocasuale di matrice e vettore */
        srand(time(NULL));
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                matrix[i * col + j] = ((double)rand() * 10 / (double)RAND_MAX) - 5; //reali nell'intervallo (-5,+5)
            }   
        }

        for (int i = 0; i < col; i++) {
            vet[i] = ((double)rand() * 10 / (double)RAND_MAX) - 5;
        }

        /* Calcolo numero di colonne per ogni processore */
        local_col = col / np;
    }
    /* Invio in broadcast del numero di righe, colonne e colonne locali */
    MPI_Bcast(&row, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&col, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&local_col, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Allocazione strutture locali ad ogni processore */
    local_matrix = (double*)malloc(row * local_col * sizeof(double));
    local_vet = (double*)malloc(local_col * sizeof(double));
    local_vet_prod = (double*)malloc(row * sizeof(double));

    /* Suddivisione per colonne delle matrice */
    for (int i = 0; i < row; i++) {
        MPI_Scatter(&matrix[i*col], local_col, MPI_DOUBLE, &local_matrix[i*local_col], local_col, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    /* Suddivisione del vettore per righe */
    MPI_Scatter(vet, local_col, MPI_DOUBLE, local_vet, local_col, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Sincronizzazione dei processori e calcolo tempo di inizio */
    MPI_Barrier(MPI_COMM_WORLD);
	T_inizio = MPI_Wtime();

    /* Prodotto matrice-vettore locale */
    for (int i = 0; i < row; i++) {
        local_vet_prod[i] = 0;
        for (int j = 0; j < local_col; j++) {
            local_vet_prod[i] += local_matrix[i*local_col+j] * local_vet[j];
        }
    }

    /* Riduzione con somma dei vettori prodotto locali */
    MPI_Reduce(local_vet_prod, vet_prod, row, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    /* Sincronizzazione dei processori e calcolo tempo di fine */
    MPI_Barrier(MPI_COMM_WORLD);
	T_fine = MPI_Wtime() - T_inizio;

    /* Calcolo del tempo totale di esecuzione */
	MPI_Reduce(&T_fine,&T_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

    /* Il processore root stampa i risultati */
    if (rank==0) {
        if (row <= 10 && col <= 10) {
            printf("\nMatrice:\n");
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < col; j++) {
                    printf("%f\t", matrix[i*col+j]);
                }
                printf("\n");
            }

            printf("\nVettore:\n");
            for (int i = 0; i < col; i++) {
                printf("%f\n", vet[i]);
            }
            
            printf("\nVettore Prodotto:\n");
            for (int i = 0; i < row; i++) {
                printf("%f\n", vet_prod[i]);
            }
        }
        printf("\nProcessori: %d\n", np);
        printf("Dimensioni matrice: %dx%d\n", row, col);
        printf("Lunghezza vettore: %d\n", col);
        printf("Tempo esecuzione paralleo: %f sec\n", T_max);
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
