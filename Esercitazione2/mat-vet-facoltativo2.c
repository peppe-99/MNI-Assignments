#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<mpi.h>

int main(int argc, char *argv[]) {
    
    int np, rank;
    int row, col, local_row;
    int *row_to_send, *displs;
    double *matrix, *vet, *vet_prod;
    double *local_matrix, *local_vet_prod;

    double T_inizio, T_fine, T_max;

    MPI_Datatype row_type;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

        /* Calcolo ed invio del numero di righe locali ad ogni processore */
        row_to_send = (int*)malloc(np * sizeof(int));
        displs = (int*)malloc(np * sizeof(int));
        for (int i = 0; i < np; i++) {
            row_to_send[i] = (row % np) > i ? (row / np) + 1 : (row / np);
            displs[i] = (i==0) ? 0 : displs[i-1] + row_to_send[i-1];

            MPI_Send(&row_to_send[i], 1, MPI_INT, i, i, MPI_COMM_WORLD);
        }
    }
    /* Ogni processore riceve il numero di righe locali */
    MPI_Recv(&local_row, 1, MPI_INT, 0, rank, MPI_COMM_WORLD, NULL);
    /* Invio in broadcast del numero colonne */
    MPI_Bcast(&col, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Allocazione strutture locali ad ogni processore */
    local_matrix = (double*)malloc(local_row * col * sizeof(double));
    local_vet_prod = (double*)malloc(local_row * sizeof(double));
    if (rank != 0) vet = (double*)malloc(col * sizeof(double));

    /* Definiamo il tipo riga */
    MPI_Type_vector(1, col, 1, MPI_DOUBLE, &row_type);
    MPI_Type_commit(&row_type);

    /* Suddivisione della matrice in blocchi di righe */
    MPI_Scatterv(matrix, row_to_send, displs, row_type, local_matrix, local_row, row_type, 0, MPI_COMM_WORLD);

    /* Invio in brodcast del vettore */
    MPI_Bcast(vet, col, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Sincronizzazione dei processori e calcolo tempo di inizio */
    MPI_Barrier(MPI_COMM_WORLD);
	T_inizio = MPI_Wtime();

    /* Prodotto matrice-vettore locale */
    for (int i = 0; i < local_row; i++) {
        local_vet_prod[i] = 0;
        for (int j = 0; j < col; j++) {
            local_vet_prod[i] += local_matrix[i*col+j] * vet[j];
        }
    }

    /* Il processore master raccoglie i risultati parziali */
    MPI_Gatherv(local_vet_prod, local_row, MPI_DOUBLE, vet_prod, row_to_send, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Calcolo tempo di fine */
    MPI_Barrier(MPI_COMM_WORLD);
	T_fine=MPI_Wtime()-T_inizio;

    /* Calcolo del tempo totale di esecuzione*/
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

    /* free della memoria */
    if (rank == 0) {
        free(matrix);
        free(vet_prod);
        free(row_to_send);
        free(displs);
    }
    free(vet);
    free(local_matrix);
    free(local_vet_prod);
    
    return 0;
}
