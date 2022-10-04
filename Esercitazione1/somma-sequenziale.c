#include<stdio.h>
#include<stdlib.h>
#include<time.h>

int main(int argc, char *argv[]) {
    
    int n;
    float tempo_speso, somma = 0.0;
    float* elementi;


    /* Input: numero degli elementi da sommare */
    printf("Inserire numero elementi: ");
    scanf("%d", &n);
 
    /* Allochiamo memoria per il vettore degli elementi da sommare */
    elementi = (float*) malloc(sizeof(float) * n);

    /* Inizializzazione del vettore con numeri reali pseudocasuali wsull'intervallo (-5, +5) */
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        elementi[i] =  ((float)rand() * 10 / (float)RAND_MAX) - 5;
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
    tempo_speso = (float)(fine - inizio) / CLOCKS_PER_SEC;    
    
    printf("Somma: %f\n", somma);
    printf("Tempo: %f sec per %d numeri reali\n", tempo_speso, n);

    return 0;
}
