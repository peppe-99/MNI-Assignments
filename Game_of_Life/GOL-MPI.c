#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<mpi.h>

#define SEED 28

/* Codici ASCII */
#define LIVE "\xF0\x9F\x91\xBE"
#define DEATH "\xF0\x9F\x94\xB2"

/* Funzione per lo swap delle matrici di lettura e scrittura */
void swap(int **current_matrix, int **new_matrix);

/* Funzione per il calcolo dei vicini vivi */
int neighbors_alive(int i, int j, int rows, int cols, int *process_matrix, int *top_row, int *bottom_row);

/* Funzione per aggiornare lo stato delle celle */
void update_generation(int from, int to, int rows, int cols, int *process_matrix, int *new_process_matrix, int *top_row, int *bottom_row);

int main(int argc, char **argv) {
    
    int np, rank, rows, cols, generations;
    int local_size, local_rows;
    int start_row, end_row;

    double T_start, T_end, T_max;
    
    /* Richieste per send e receive asincrone */
    MPI_Request send_first_row, send_last_row, receive_next_row, receive_prev_row;

    /* Inizializzazione dell'ambiente MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    /* Controllo sul numero di parametri */
    if (argc != 4) {
        if (rank == 0) { fprintf(stderr, "Input: righe, colonne e generazioni\n"); }
        MPI_Finalize();
        return 0;
    }

    /* Input: dimensioni matrice e numero di generazioni */
    rows = atoi(argv[1]);
    cols = atoi(argv[2]);
    generations = atoi(argv[3]);

    /* Controllo se il numero delle righe è maggiore del numero di processori */
        if (rows < np) {
        if (rank == 0) fprintf(stderr, "Il numero di righe deve essere almeno pari al numero di processori usati\n");
        MPI_Finalize();
        return 0;
    }

    /* Matrice generale */
    int *matrix = (int*)malloc((rows * cols) * sizeof(int));

    /* send_counts contiene quanti elementi riceverà ogni processore */
    int *send_counts = (int*)malloc(np * sizeof(int));

    /* Displacements */
    int *displs = (int*)malloc(np * sizeof(int));

    /* Calcolo di quante righe inviare ad ogni processore e dei displacements */
    for (int i = 0; i < np; i++) {
        send_counts[i] = (i < (rows % np)) ? ((rows / np) + 1) * cols : (rows / np) * cols;
        displs[i] = (i == 0) ? 0 : displs[i-1] + send_counts[i-1];
    }

    /* Dimensione matrice locale e numero di righe locali ad ogni processore */
    local_size = send_counts[rank];
    local_rows = local_size / cols;

    /* Matrice locale di lettura (da cui leggiamo lo stato attuale) */
    int *process_matrix = (int*)malloc(local_size * sizeof(int));

    /* Matrice locale di scrittura (in cui scriviamo lo stato seguente) */
    int *new_process_matrix = (int*)malloc(local_size * sizeof(int)); 

    /* Prima riga locale da inviare al processore precedente (se esiste) */
    int *top_row = NULL;
    if (rank > 0) {
        top_row = (int*)malloc(cols * sizeof(int));
    }

    /* Ultima riga locale da inviare al processore successivo (se esiste) */
    int *bottom_row = NULL;
    if (rank < np-1) {
        bottom_row = (int*)malloc(cols * sizeof(int));
    }

    /* Il master genera la matrice di partenza con la generazione iniziale */
    if (rank == 0) {
        srand(SEED);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i * cols + j] = rand() % 2;
            }
        }

        /* stampa matrice generazione iniziale */
        if (rows <= 10 && cols <= 10) {
            printf("Generazione Iniziale:\n");
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    (matrix[i * cols + j] == 1) ? printf(LIVE) : printf(DEATH);
                }
                printf("\n");
            }
        }
    }

    /* Invio ad ogni processo, master compreso, la sua sottomatrice (blocco di righe) */
    MPI_Scatterv(matrix, send_counts, displs, MPI_INT, process_matrix, local_size, MPI_INT, 0, MPI_COMM_WORLD);

    /* Sincronizzazione dei processori e calcolo tempo di inizio */
    MPI_Barrier(MPI_COMM_WORLD);
	T_start = MPI_Wtime();

    /* Simulazione generazioni */
    for (int generation = 0; generation < generations; generation++) {

        /* Tutti i processori tranne quello con rank 0 (master): 
            - inviano al precedente la propria prima riga 
            - richiedono al precedente la sua ultima riga */
        if (rank > 0) {
            MPI_Isend(&process_matrix[0], cols, MPI_INT, rank-1, rank-1, MPI_COMM_WORLD, &send_first_row);
            MPI_Irecv(top_row, cols, MPI_INT, rank-1, rank, MPI_COMM_WORLD, &receive_prev_row);
        }

        /* Tutti i processori tranne quello con rank np-1 (l'ultimo): 
            - inviano al successivo la propria ultima riga 
            - richiedono al successivo la sua prima riga  */
        if (rank < np-1) {
            MPI_Isend(&process_matrix[(local_rows-1) * cols], cols, MPI_INT, rank+1, rank+1, MPI_COMM_WORLD, &send_last_row);
            MPI_Irecv(bottom_row, cols, MPI_INT, rank+1, rank, MPI_COMM_WORLD, &receive_next_row);
        }

        /* Ogni processore inizia a computare le righe non vincolate (se ci sono) */
        start_row = (rank == 0) ? 0 : 1;
        end_row = (rank == np-1) ? local_rows : local_rows-1;
        update_generation(start_row, end_row, local_rows, cols, process_matrix, new_process_matrix, NULL, NULL);

        /* Se la sottomatrice di un processore è composta da una sola riga allora è doppiamente vincolata */
        if (local_rows == 1) {
            if (rank > 0) MPI_Wait(&receive_prev_row, MPI_STATUS_IGNORE);
            if (rank < np-1) MPI_Wait(&receive_next_row, MPI_STATUS_IGNORE);

            start_row = 0;
            update_generation(start_row, start_row+1, local_rows, cols, process_matrix, new_process_matrix, top_row, bottom_row);
        }

        /* Altrimenti abbiamo la prima e/o l'ultima riga vincolate */
        else {
            if (rank > 0) {
                MPI_Wait(&receive_prev_row, MPI_STATUS_IGNORE);
                
                start_row = 0;
                update_generation(start_row, start_row+1, local_rows, cols, process_matrix, new_process_matrix, top_row, bottom_row);
            }
            
            if (rank < np-1) {
                MPI_Wait(&receive_next_row, MPI_STATUS_IGNORE);
                
                start_row = local_rows-1;
                update_generation(start_row, start_row+1, local_rows, cols, process_matrix, new_process_matrix, top_row, bottom_row);
            }           
        }
        /* swap delle matrici */
        swap(&process_matrix, &new_process_matrix);
    }

    /* Otteniamo la generazione finale */
    MPI_Gatherv(process_matrix, local_size, MPI_INT, matrix, send_counts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    /* Tempo di fine */
    MPI_Barrier(MPI_COMM_WORLD);
    T_end = MPI_Wtime();

    /* Calcolo del tempo totale di esecuzione */
	MPI_Reduce(&T_end, &T_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* Stampa dei risultati */
    if (rank == 0) {
        if (rows <= 10 && cols <= 10) {
            printf("\nGenerazione %d:\n", generations);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    (matrix[i * cols + j] == 1) ? printf(LIVE) : printf(DEATH);
                }
                printf("\n");
            }
        }
        printf("\nTempo di esecuzion con %d processori: %fs\n", np, T_max);
    }

    /* free della memoria */ 
    free(matrix);
    free(displs);
    free(send_counts);
    free(process_matrix);
    free(new_process_matrix);
    if (rank > 0) {
        free(top_row);
    }
    if (rank < np-1) {
        free(bottom_row);
    }

    MPI_Finalize();

    return 0;
}

void swap(int **current_matrix, int **new_matrix) {
    int *temp = *current_matrix;
    *current_matrix = *new_matrix;
    *new_matrix = temp;
}

int neighbors_alive(int i, int j, int rows, int cols, int *process_matrix, int *top_row, int *bottom_row) {
    
    /* vicini superiori */
    int top = 0, tl_corner = 0, tr_corner = 0;
    if (i > 0) {
        top = process_matrix[(i-1) * cols + j];        
        if (j > 0) {
            tl_corner = process_matrix[(i-1) * cols + (j-1)];
        }
        if (j < cols-1) {
            tr_corner = process_matrix[(i-1) * cols + (j+1)];
        }
    }
    /* i vicini superiori fanno parte della top_row */
    else if(top_row != NULL) {
        top = top_row[j];
        if (j > 0) {
            tl_corner = top_row[j-1];
        }
        if (j < cols-1) {
            tr_corner = top_row[j+1];
        }
    } 
    /* else: i vicini superiori non eisstono */

    /* vicini laterali */
    int left = (j > 0) ? process_matrix[i * cols + (j-1)] : 0;
    int right = (j < cols-1) ? process_matrix[i * cols + (j+1)] : 0;

    /* vicini inferiori */
    int bottom = 0, bl_corner = 0, br_corner = 0;
    if (i < rows-1) {
        bottom = process_matrix[(i+1) * cols + j];
        if (j > 0) {
            bl_corner = process_matrix[(i+1) * cols + (j-1)];
        }
        if (j < cols-1) {
            br_corner = process_matrix[(i+1) * cols + (j+1)];
        }
    }
    /* i vicini inferiori fanno parte della bottom_row */
    else if(bottom_row != NULL) {
        bottom = bottom_row[j];
        if (j > 0) {
            bl_corner = bottom_row[j-1];
        }
        if (j < cols-1) {
            br_corner = bottom_row[j+1];
        }
    }
    /* else: i vicini inferiori non esistono */

    return top + left + right + bottom + tl_corner + tr_corner + bl_corner + br_corner;
}

void update_generation(int from, int to, int rows, int cols, int *process_matrix, int *new_process_matrix, int *top_row, int *bottom_row) {
    for (int i = from; i < to; i++) {
        for (int j = 0; j < cols; j++) {
            int alives = neighbors_alive(i, j, rows, cols, process_matrix, top_row, bottom_row);

            if (process_matrix[i * cols + j] == 1 && (alives == 2 || alives == 3)) {
                new_process_matrix[i * cols + j] = 1;
            }
            else if (process_matrix[i * cols + j] == 0 && alives == 3) {
                new_process_matrix[i * cols + j] = 1;
            }
            else new_process_matrix[i * cols + j] = 0;
        }
    }
}

