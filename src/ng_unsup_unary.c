#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "network.h"
#include "network_trainer.h"

#define PRERUN_INDICES 5
#define PRERUN_MULT 2
#define SMALL_ENOUGH_RATE 0.005

int number_to_train = 0;


void training_session(values_t* network, connections_t* connections, FILE* training_data, double learning_speed, int start_with_print){
    long labels[1000];
    double content[1000 * network->layer_sizes[0]];

    for(int i = 0; i < 1000; i++){
        if(!fread(labels + i, sizeof(long), 1, training_data)) exit(EXIT_FAILURE);
        if(!fread(content + (i*network->layer_sizes[0]), sizeof(double) * network->layer_sizes[0], 1, training_data)) exit(EXIT_FAILURE);
    }

    double desired_output[network->layer_sizes[network->layer_sizes_length - 1]];
    
    for(int j = 0; j < 1000; j ++){
        memset(desired_output, 0, network->layer_sizes[network->layer_sizes_length - 1] * sizeof(double));
        
        int repeats = 1;
        if(labels[j] == number_to_train){
            desired_output[0] = 1;
            desired_output[1] = 0;
            //Since it is a unary run, more repeats have to be conducted in order not to sway the balance too far in the direction of not
            repeats = 9;
        } else {
            desired_output[0] = 0;
            desired_output[1] = 1;
        }
        
        int print = (start_with_print && j == 0) ? 1 : 0;

        for(int i = 0; i < repeats; i++){
            gradient_decent(network, connections, content + j * network->layer_sizes[0], desired_output, learning_speed, print);
        }
        apply_connections(network, connections);

        if(print){
            printf("outputs for in %ld:\n", labels[j]);
            for(int i = 0; i < network->layer_sizes[network->layer_sizes_length - 1]; i++){
                printf("%d:%f; ", i, network->layers[network->layer_sizes_length - 1][i]);
            }
            printf("\n");
        }
    }
}

double calc_trainingRate(double base, double reduc, int supersession){
    if(supersession < PRERUN_INDICES){
        return SMALL_ENOUGH_RATE * pow(PRERUN_MULT, (double)supersession);
    }

    return base + (reduc * supersession);
}

int main (int argc, char* argv[]){
    values_t *network = init_values((int[]){300, 150, 100, 2}, 4);
    connections_t *connections = init_connections((int[]){300, 150, 100, 2}, 4);

    number_to_train = strtol(argv[1], NULL, 10);

    double training_rate = strtod(argv[2], NULL);
    double reduction = strtod(argv[3], NULL);
    
    randomise_all_connections(connections, 10);

    for(int supersession = 0; supersession < 600; supersession++){
        FILE *training_data = fopen("data/unsup/bin_mnist_train_300", "r");

        printf("##########\nsupersession: %d\ntraining rate: %f\n##########\n", supersession, calc_trainingRate(training_rate, reduction, supersession));

        for(int i = 0; i < 50; i++){
            int print = (i==0 || i==49) ? 1 : 0;
            training_session(network, connections, training_data, calc_trainingRate(training_rate, reduction, supersession), print);
        }

        fclose(training_data);

        char outname[64];
        snprintf(outname, 64, "%s%d", argv[4], supersession);
        save_in_file(network, connections, outname);
    }

    return 0;
}