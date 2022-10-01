#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int main(int argc, char **argv) {
    
    int np, rank;
    int dim, dim_locale;
    double *elementi, *elementi_locali;
    double somma=0.0, somma_locale=0.0;
    double inizio, fine;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    if (rank == 0) {
        printf("Inserire numero elementi: ");
        fflush(stdout);
        scanf("%d", &dim);

        elementi = (double*)malloc(dim * sizeof(double));

        srand(time(NULL)); 
        for (int i = 0; i < dim; i++) {
            elementi[i] =  ((double)rand() * 10 / (double)RAND_MAX) - 5;
            printf("%f ", elementi[i]);
        }
        printf("\n\n");
    }

    // invio in brodcast il numero di elementi
    MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    int* send_counts = (int*)malloc(np * sizeof(int));
    int* displs = (int*)malloc(np * sizeof(int));

    for (int i = 0; i < np; i++) {
        send_counts[i] = i < (dim % np) ? (dim/np)+1 : dim/np;
        displs[i] = (i==0) ? 0 : displs[i-1] + send_counts[i-1];
    }
    dim_locale = send_counts[rank];

    elementi_locali = (double*)malloc(dim_locale * sizeof(double));


    MPI_Scatterv(elementi, send_counts, displs, MPI_DOUBLE, elementi_locali, dim_locale, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < dim_locale; i++) {
        printf("%f ", elementi_locali[i]);
    }
    printf("\n\n");

    MPI_Finalize();

    return 0;
}
