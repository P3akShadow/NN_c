#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "network.h"
#include "network_trainer.h"

#define PRERUN_INDICES 9
#define PRERUN_MULT 1.2
#define SMALL_ENOUGH_RATE 0.1


void training_session(values_t* network, connections_t* connections, FILE* training_data, double learning_speed, int start_with_print){
    long labels[1000];
    double content[1000 * 28*28];

    char buffer[1<<13];
    if(!fgets(buffer, 1<<13, training_data)) exit(EXIT_FAILURE);

    for(int i = 0; i < 1000; i++){
        if(!fgets(buffer, 1<<13, training_data)) exit(EXIT_FAILURE);
        labels[i] = strtol(strtok(buffer, ","), NULL, 10);

        for(int j = 0; j < 28*28; j++){
            long intermediate = strtol(strtok(NULL, ","), NULL, 10);
            content[i*28*28+j] = ((double) intermediate) / 256;
        }
    }

    double desired_output[10];
    
    for(int j = 0; j < 1000; j ++){
        memset(desired_output, 0, 10 * sizeof(double));
        desired_output[labels[j]] = 1;

        int print = (start_with_print && j == 0) ? 1 : 0;
        gradient_decent(network, connections, content + j * 28 * 28, desired_output, learning_speed, print);
        apply_connections(network, connections);
    }
}

double calc_trainingRate(double base, double reduc, int supersession){
    if(supersession < PRERUN_INDICES){
        return SMALL_ENOUGH_RATE * pow(PRERUN_MULT, (double)supersession);
    }

    return base * pow(reduc, (double)supersession);
}

int main (int argc, char* argv[]){
    values_t *network = init_values((int[]){28*28, 300, 100, 10}, 4);
    connections_t *connections = init_connections((int[]){28*28, 300, 100, 10}, 4);

    double training_rate = strtod(argv[1], NULL);
    double reduction = strtod(argv[2], NULL);
    
    randomise_all_connections(connections, 10);

    for(int supersession = 0; supersession < 200; supersession++){
        FILE *training_data = fopen("data/mnist_train.csv", "r");


        printf("##########\nsupersession: %d\ntraining rate: %f\n##########\n", supersession, calc_trainingRate(training_rate, reduction, supersession));

        char buffer[1<<13];
        if(!fgets(buffer, 1<<13, training_data)) exit(EXIT_FAILURE);
        for(int i = 0; i < 50; i++){
            printf("\tsession: %d\n", i);

            int print = (i==0 || i==49) ? 1 : 0;
            training_session(network, connections, training_data, calc_trainingRate(training_rate, reduction, supersession), print);
        }

        fclose(training_data);

        char outname[64];
        snprintf(outname, 64, "%s%d", argv[3], supersession);
        save_in_file(network, connections, outname);
    }

    return 0;
}