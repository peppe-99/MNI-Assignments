#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int main(int argc, char **argv) {
    
    int np, rank;
    int dim, dim_locale, passi=0, p;
    int *displs, *send_counts, *potenze;
    float *elementi, *elementi_locali;
    float somma_locale=0.0, oracolo=0.0, somma_parziale;
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
        elementi = (float*)malloc(dim * sizeof(float));
        srand((unsigned int) time(0)); 
        for (int i = 0; i < dim; i++) {
            elementi[i] =  ((float)rand() * 10 / (float)RAND_MAX) - 5;
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

    elementi_locali = (float*)malloc(dim_locale * sizeof(float));

    MPI_Scatterv(elementi, send_counts, displs, MPI_FLOAT, elementi_locali, dim_locale, MPI_FLOAT, 0, MPI_COMM_WORLD);

    /* Sincronizzazione dei processori e calcolo tempo di inizio */
    MPI_Barrier(MPI_COMM_WORLD);
    inizio = MPI_Wtime();

    /* Calcolo del numero di passi temporale, ovvero log_2(np) */
    for (p = np; p != 1; p=p>>1) {
        passi++;
    }

    /* Creazione del vettore con le potenze di 2 */
    potenze = (int*) malloc((passi+1) * sizeof(int));
	for(int i=0; i <= passi; i++) {
		potenze[i] = p<<i;
	}
    /* calcolo delle somme parziali e combinazioen dei risultati parziali */
    for (int i = 0; i < dim_locale; i++) {
            somma_locale += elementi_locali[i];
    }
    for (int k = 0; k < passi; k++) {
        /* se rank % 2^{k+1} == 2^k allora P_rank invia al processore con id = rank - 2^k */
        if ((rank % potenze[k+1]) == potenze[k]) {
            MPI_Send(&somma_locale, 1, MPI_FLOAT, rank-potenze[k], rank-potenze[k], MPI_COMM_WORLD);
        }
        /* altrimetni, se rank % 2^{k+1} == 0 allora P_rank riceve dal processore con id = rank + 2^k */
        else if ((rank % potenze[k+1]) == 0) {
            MPI_Recv(&somma_parziale, 1, MPI_FLOAT, rank+potenze[k], rank, MPI_COMM_WORLD, NULL);
            somma_locale+=somma_parziale;
        }
        // altrimenti, il processore ha completato il suo lavoro nel precedente passo temporale
    }

    /* Sincronizzazione dei processori e calcolo del tempo di fine */
    MPI_Barrier(MPI_COMM_WORLD);
    fine_locale = MPI_Wtime() - inizio;

    /* Calcolo tempo totale di esecuzione */
    MPI_Reduce(&fine_locale, &fine, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        if (dim <= 10) {
            printf("\nNumeri: ");
            for (int i = 0; i < dim; i++) {
                printf("%f ", elementi[i]);
            }
        }
        printf("\n\nSomma parallela: %f\n", somma_locale);
        printf("\nTempo locale P0: %lf sec\n", fine_locale);
        printf("\nTempo totale: %lf sec\n", fine);
    }

    MPI_Finalize();

    return 0;
}
