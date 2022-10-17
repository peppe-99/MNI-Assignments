/**
 *  Studente: Giuseppe Cardaropoli
 *  Matricola: 0522501310
 *  Algoritmo: prodotto matrice-vettore sequenziale
 */
#include<stdio.h>
#include<stdlib.h>
#include<time.h>

int main(int argc, char *argv[]) {
    
    int row, col;
    float tempo;
    double *matrix, *vet, *vet_prod;

    /* Input: dimensioni della matrice */
    printf("Inserire numero di righe: ");
    scanf("%d", &row);
    printf("Inserire numero di colonne: ");
    scanf("%d", &col);

    /* Controllo attivazione stampe */


    /* Allocazione delle strutture dati */
    matrix = (double*)malloc(row * col * sizeof(double));
    vet = (double*)malloc(col * sizeof(double));
    vet_prod  = (double*)malloc(row * sizeof(double));

    /* Inizializzazione pseudocasuale di matrice e vettore */
    srand(time(NULL));
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            matrix[i * col + j] = ((double)rand() * 10 / (double)RAND_MAX) - 5; //reali nell'intervallo (-5,+5)
        }
    }

    for (int i = 0; i < col; i++) {
        vet[i] = ((double)rand() * 10 / (double)RAND_MAX) - 5;
    }

    /* Tempo di inizio */
    clock_t inizio = clock();    

    /* Calcolo prodotto matrice-vettore */
    for (int i = 0; i < row; i++) {
        vet_prod[i] = 0;
        for (int j = 0; j < col; j++) {
            vet_prod[i] += matrix[i * col + j] * vet[j];
        }
    }

    /* Tempo di fine*/
    clock_t fine = clock();    

    /* Calcolo tempo di esecuzione */
    tempo = (float)(fine - inizio) / CLOCKS_PER_SEC;    

    /* Stampa di dati e risultati */
    if (row <= 10 && col <=10) {
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
    printf("\nTempo di esecuzione sequenziale: %f sec\n", tempo);

    /* free della memoria */
    free(matrix);
    free(vet);
    free(vet_prod);

    return 0;
}
