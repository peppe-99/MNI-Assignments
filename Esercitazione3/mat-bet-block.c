/**
 *  Studente: Giuseppe Cardaropoli
 *  Matricola: 0522501310
 *  Algoritmo: prodotto matrice-vettore parallelo
 *  Strategia: divisione della matrice in blocchi di righe e colonne
 */
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<mpi.h>

int main(int argc, char *argv[]) {

    /* Parametri generali */
    int rank, np;    
    int nrow, ncol, local_row, local_col;

    /* Parametri griglia cartesiana 2D */
    int dims[2], period[2] = {0,1}, reorder = 1;
    int rank2D, coords2D[2];

    /* Parameteri sottogriglie 1D (colonna e riga) */
    int rankCol, rankRow, coords1DCol[1], coords1DRow[1];
    int belongs[2];

    /* Strutture dati */
    double *matrix, *vet, *vet_prod, *oracolo;
    double *local_matrix, *local_vet, *local_prod;
    double *matrix_temp, *prod_temp;

    /* Comunicatori */
    MPI_Comm comm2D, comm_col, comm_row;

    /* Tipo blocco */
    MPI_Datatype block, blocktype;

    /* Variabii per il tempo  d'esecuzione */
    double T_inizio, T_fine, T_max;

    /* Inizializzazione ambiente MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    /* Il processore 0 ottiene le dimensioni e inizializza le strutture dati */
    if (rank == 0) {
        printf("Inserire numero di righe: ");
        fflush(stdout);
        scanf("%d", &nrow);

        printf("Inserire numero di colonne: ");
        fflush(stdout);
        scanf("%d", &ncol);

        /* Allocazione delle strutture dati */
        matrix = (double*)malloc(nrow * ncol * sizeof(double));
        vet = (double*)malloc(ncol * sizeof(double));
        vet_prod = (double*)malloc(nrow * sizeof(double));
        oracolo = (double*)malloc(nrow * sizeof(double));

        /* Inizializzazione pseudocasuali di matrice e vettore */
        srand(time(NULL));
        for (int i = 0; i < nrow; i++) {
            for (int j = 0; j < ncol; j++) {
                //matrix[i * ncol + j] = (double) i * ncol + j;
                matrix[i * ncol + j] = ((double)rand() * 10 / (double)RAND_MAX) - 5; //reali nell'intervallo (-5,+5)
            }
        }
        for (int i = 0; i < ncol; i++) {
            //vet[i] = (double) i;
            vet[i] = ((double)rand() * 10 / (double)RAND_MAX) - 5; //reali nell'intervallo (-5,+5)
        }

        /* Calcolo dell'oracolo */
        for (int i = 0; i < nrow; i++) {
            oracolo[i] = 0;
            for (int j = 0; j < ncol; j++) {
                oracolo[i] += matrix[i*ncol+j] * vet[j];
            }
        }
    }
    /* Invio in broadcast del numero di righe e colonne */
    MPI_Bcast(&nrow, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ncol, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Calcolo e controllo le dimensioni della griglia di processori */
    dims[0] = np/2; dims[1] = np/dims[0];
    if (rank == 0) {
        if ((nrow % dims[0]) != 0 || (ncol % dims[1]) != 0) {
            printf("\nLe dimensioni della matrice devono essere multiple del numero di processori\n\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    /* Creazione della topologia cartesiana 2D */
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, period, reorder, &comm2D);
    MPI_Comm_rank(comm2D, &rank2D);
    MPI_Cart_coords(comm2D, rank2D, 2, coords2D);

    /* Creazione comunicatore sottogrigle di colonne 1D */
    belongs[0] = 1; belongs[1] = 0;
    MPI_Cart_sub(comm2D, belongs, &comm_col);
    MPI_Comm_rank(comm_col, &rankCol);
    MPI_Cart_coords(comm_col, rankCol, 1, coords1DCol);

    /* Creazione comunicatore sottogriglie di righe 1D */
    belongs[0] = 0; belongs[1] = 1;
    MPI_Cart_sub(comm2D, belongs, &comm_row);
    MPI_Comm_rank(comm_row, &rankRow);
    MPI_Cart_coords(comm_row, rankRow, 1, coords1DRow);

    /* Con questa barrier mi assicuro che ogni processore abbia ottenuto le proprie coordinate */
    MPI_Barrier(MPI_COMM_WORLD);
        
    /* Ogni processore calcola le proprie dimensioni locali */
    local_row = nrow / dims[0];
    local_col = ncol / dims[1];

    /* Allocazione strutture dati locali */
    local_matrix = (double*)malloc(local_row * local_col * sizeof(double));
    matrix_temp = (double*)malloc(local_row * ncol * sizeof(double));
    local_vet = (double*)malloc(local_col * sizeof(double));
    local_prod = (double*)malloc(local_row * sizeof(double));
    prod_temp = (double*)malloc(local_row * sizeof(double));

    /* SUDDIVISIONE DEL VETTORE */
    if (coords2D[0] == 0) {
        /* P(0,0) suddivide il vettore lungo la prima riga di processori */
        MPI_Scatter(vet, local_col, MPI_DOUBLE, local_vet, local_col, MPI_DOUBLE, 0, comm_row);
    }
    /* Il primo processore di ogni colonna invia in brodcast, lungo la propria riga di processori, il vettore */
    MPI_Bcast(local_vet, local_col, MPI_DOUBLE, 0, comm_col);

    /* SUDDIVISIONE DELLA MATRICE */
    if (coords2D[1] == 0) {
        /* P(0,0) suddivide la matrice in blocchi di righe lungo la prima colonna di processori */
        MPI_Scatter(matrix, local_row * ncol, MPI_DOUBLE, matrix_temp, local_row * ncol, MPI_DOUBLE, 0, comm_col);
    }
    /* Definiamo il tipo blocco */
    MPI_Type_vector(local_row, local_col, ncol, MPI_DOUBLE, &block);
    MPI_Type_commit(&block);
    MPI_Type_create_resized(block, 0, local_col*sizeof(double), &blocktype);
    MPI_Type_commit(&blocktype);

    /* Il primo processore di ogni riga suddivide, lungo la propria riga di processori, il blocco di righe in blocchi di righe e colonne */
    MPI_Scatter(matrix_temp, 1, blocktype, local_matrix, local_row * local_col, MPI_DOUBLE, 0, comm_row);

    /* Calcolo tempo di inizio */
    MPI_Barrier(MPI_COMM_WORLD);
	T_inizio = MPI_Wtime();

    /* Calcolo prodotto mat-vet locale */        
    for (int i = 0; i < local_row; i++) {
        local_prod[i] = 0;
        for (int j = 0; j < local_col; j++) {
            local_prod[i] += local_matrix[i * local_col + j] * local_vet[j];
        }          
    }

    /* Somme lungo le righe dei prodotti parziali */
    MPI_Reduce(local_prod, prod_temp, local_row, MPI_DOUBLE, MPI_SUM, 0, comm_row);

    /* Combinazione lungo la prima colonna */
    if (coords2D[1] == 0) {
        MPI_Gather(prod_temp, local_row, MPI_DOUBLE, vet_prod, local_row, MPI_DOUBLE, 0, comm_col);
    }

    /* Sincronizzazione dei processori e calcolo tempo di fine */
    MPI_Barrier(MPI_COMM_WORLD);
	T_fine = MPI_Wtime() - T_inizio;

    /* Calcolo del tempo totale di esecuzione */
	MPI_Reduce(&T_fine,&T_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

    /* Stampa dei risultati */
    if (rank == 0) {
        if (nrow <= 10 && ncol <= 10) {
            printf("\nMatrice\n");
            for (int i = 0; i < nrow; i++) {
                for (int j = 0; j < ncol; j++) {
                    printf("%f\t", matrix[i * ncol + j]);
                }
                printf("\n");
            }

            printf("\nVettore\n");
            for (int i = 0; i < ncol; i++) {
                printf("%f\n", vet[i]);
            }

            printf("\nVettore Prodotto\n");
            for (int i = 0; i < nrow; i++) {
                printf("%f\n", vet_prod[i]);
            }

            printf("\nOracolo\n");
            for (int i = 0; i < nrow; i++) {
                printf("%f\n", oracolo[i]);
            }
        }
        printf("\nProcessori: %d\n", np);
        printf("Dimensioni matrice: %dx%d\n", nrow, ncol);
        printf("Lunghezza vettore: %d\n", ncol);
        printf("Tempo esecuzione paralleo: %f sec\n", T_max);
    }

    MPI_Finalize();

    /* free della memoria */
    if (rank == 0) {
        free(matrix);
        free(vet);
        free(vet_prod);
        free(oracolo);
    }
    free(local_matrix);
    free(matrix_temp);
    free(local_vet);
    free(local_prod);
    free(prod_temp);

    return 0;
}