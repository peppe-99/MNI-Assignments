#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>
#include<math.h>
#include<time.h>

int main(int argc, char **argv) {
    
    int np, rank;
    int dim, dim_locale, passi=0, p, comunicate_with;
    int *displs, *send_counts, *potenze;
    double *elementi, *elementi_locali;
    double somma=0.0, somma_locale=0.0, oracolo=0.0, somma_parziale;
    double inizio, fine_locale, fine;

    /* Inizializzazione ambiente MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);


    if (rank == 0) {
        /* Il processore P0 legge il numero di elementi da sommare */
        printf("Inserire numero elementi: ");
        fflush(stdout);
        scanf("%d", &dim);

        /* Inizializzazione del vettore con reali pseudocasuali nell'intervallo (-5,+5) */
        /* Viene calcolato anche un oracolo per controllare la correttezza della somma parallela */
        elementi = (double*)malloc(dim * sizeof(double));
        srand((unsigned int) time(0)); 
        for (int i = 0; i < dim; i++) {
            elementi[i] =  ((double)rand() * 10 / (double)RAND_MAX) - 5;
            oracolo += elementi[i];
        }
        printf("\nOracolo: %f\n", oracolo);
    }

    /* Invio in brodcast del numero di elementi da sommare */
    MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    /* Ripartizione equa degli elementi da sommare */
    send_counts = (int*)malloc(np * sizeof(int));
    displs = (int*)malloc(np * sizeof(int));

    for (int i = 0; i < np; i++) {
        send_counts[i] = i < (dim % np) ? (dim/np)+1 : dim/np;
        displs[i] = (i==0) ? 0 : displs[i-1] + send_counts[i-1];
    }
    dim_locale = send_counts[rank];

    elementi_locali = (double*)malloc(dim_locale * sizeof(double));

    MPI_Scatterv(elementi, send_counts, displs, MPI_DOUBLE, elementi_locali, dim_locale, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Sincronizzazione dei processori e calcolo tempo di inizio */
    MPI_Barrier(MPI_COMM_WORLD);
    inizio = MPI_Wtime();

    for (p = np; p != 1; p=p>>1) {
        passi++;
    }

    /* Creazione del vettore con le potenze di 2 */
    potenze = (int*) malloc((passi+1) * sizeof(int));
	for(int i=0; i <= passi; i++) {
		potenze[i] = p<<i;
	}

    for (int k = 0; k < passi; k++) {
        /* Ogni processore calcola la prima somma parziale */
        if (k == 0) {
            for (int i = 0; i < dim_locale; i++) {
                somma_locale+=elementi_locali[i];
            }
        }

        /* Ogni processore calcola con chi deve comunicare */
        /* se rank % 2^{k+1} < 2^k allora P_rank comunica con il processore con id = rank + 2^k */
        if (rank % potenze[k+1] < potenze[k]) {
            comunicate_with = rank+potenze[k];
        }
        /* Altrimenti P_rank comunica con il processore con id = rank - 2^k */
        else {
            comunicate_with = rank-potenze[k];
        }

        /* Scambio ed aggiornamento delle somme parziali */
        MPI_Send(&somma_locale, 1, MPI_DOUBLE, comunicate_with, comunicate_with, MPI_COMM_WORLD);
        MPI_Recv(&somma_parziale, 1, MPI_DOUBLE, comunicate_with, rank, MPI_COMM_WORLD, NULL);
        somma_locale += somma_parziale;
    }

    /* Sincronizzazione dei processori e calcolo del tempo di fine */
    MPI_Barrier(MPI_COMM_WORLD);
    fine_locale = MPI_Wtime() - inizio;

    /* Calcolo tempo totale di esecuzione */
    MPI_Reduce(&fine_locale, &fine, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    printf("Somma locale al processore P%d: %f\n", rank, somma_locale);
    if (rank == 0) {
        printf("\nTempo totale: %lf sec\n", fine);
    }

    MPI_Finalize();
    
    return 0;
}
