#include<stdio.h>
#include<stdlib.h>
#include<time.h>

int main(int argc, char *argv[]) {
    
    int n;
    double tempo_speso, somma = 0.0;
    double* elementi;


    /* Input: numero degli elementi da sommare */
    printf("Inserire numero elementi: ");
    scanf("%d", &n);
 
    /* Allochiamo memoria per il vettore degli elementi da sommare */
    elementi = (double*) malloc(sizeof(double) * n);

    /* Inizializzazione del vettore con numeri reali pseudocasuali wsull'intervallo (-5, +5) */
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        elementi[i] =  ((double)rand() * 10 / (double)RAND_MAX) - 5;
    }

    /* Calcolo tempo d'inizio */
    clock_t inizio = clock();    

    /* Somma Sequenziale */
    for (int i = 0; i < n; i++) {
        somma += elementi[i];
    }

    /* Calcolo tempo di fine */
    clock_t fine = clock();

    /* Calcolo del tempo impiegato */
    tempo_speso = (double)(fine - inizio) / CLOCKS_PER_SEC;    
    
    printf("Somma: %f\n", somma);
    printf("Tempo: %fs per %d numeri reali\n", tempo_speso, n);

    return 0;
}
