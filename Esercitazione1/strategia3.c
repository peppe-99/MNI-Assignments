#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>
#include<math.h>
#include<time.h>

int main(int argc, char **argv) {
    
    int np, rank, size;
    float* numeri, numeri_locali;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);


    if (rank==0) {
        printf("\n Inserisci il numero di elementi: \n");
        fflush(stdout);
        scanf("%d", &size);
    }
    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    numeri = (float*) calloc(size, sizeof(float));
    
    if (rank==0) {
        srand((unsigned int) time(0));
        for (int i=0; i<size; i++) {
            *(numeri+i)=fmod((double) rand(), (double) 5.0) - 2;
        }
    }

    



    //MPI_Bcast(numeri, size, MPI_FLOAT, 0, MPI_COMM_WORLD);


    MPI_Finalize();
    
    return 0;
}
