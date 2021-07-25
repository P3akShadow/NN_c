#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "network.h"
#include "network_trainer.h"

#define PRERUN_INDICES 2
#define PRERUN_MULT 1.5
#define SMALL_ENOUGH_RATE 1

void training_session(values_t** unsup_networks, connections_t** unsup_connections, FILE* training_data, 
double learning_speed, int num_of_nets, int start_with_print){
    //unneccessary
    long labels[1000];
    double content[1000 * 28*28];

    char buffer[1<<13];
    //if(!fgets(buffer, 1<<13, training_data)) exit(EXIT_FAILURE);

    for(int i = 0; i < 1000; i++){
        if(!fgets(buffer, 1<<13, training_data)) exit(EXIT_FAILURE);
        labels[i] = strtol(strtok(buffer, ","), NULL, 10);

        for(int j = 0; j < 28*28; j++){
            long intermediate = strtol(strtok(NULL, ","), NULL, 10);
            content[i*28*28+j] = ((double) intermediate) / 256;
        }
    }
    
    //for supervised
    //double desired_output[10];

    for(int j = 0; j < 1000; j ++){
        double* layer_input = content + j * unsup_networks[0]->layer_sizes[0];

        //unsupervised
        for(int k = 0; k < num_of_nets; k++){
            int print = (start_with_print && j == 0) ? 1 : 0;
            //input shall be output too
            gradient_decent(unsup_networks[k], unsup_connections[k], layer_input, layer_input, learning_speed, print);
            apply_connections(unsup_networks[k], unsup_connections[k]);

            layer_input = unsup_networks[k]->layers[1];

            if(print){
                printf("\toutput: %ld,", labels[j]);
                for(int i = 0; i < unsup_networks[k]->layer_sizes[2]; i++){
                    int output = (int)(unsup_networks[k]->layers[unsup_networks[k]->layer_sizes_length - 1][i] * 256);
                    printf("%d,", output);
                }
                printf("\n");
            }
        }
    }
}

double calc_trainingRate(double base, double reduc, int supersession){
    if(supersession < PRERUN_INDICES){
        return SMALL_ENOUGH_RATE * pow(PRERUN_MULT, (double)supersession);
    }

    return base * pow(reduc, (double)supersession);
}

int main (int argc, char* argv[]){
    //this is the skeleton for the whole unsupervised stack
    values_t *unsup_tester_n = init_values((int[]){784, 500, 300, 500, 784}, 5);
    connections_t *unsup_tester_c = init_connections((int[]){784, 500, 300, 500, 784}, 5);
    randomise_all_connections(unsup_tester_c, 10);

    int unsupervised_layers = 2;
    
    //this is are the different unsupervised training instances
    values_t *unsupervised_networks[unsupervised_layers];
    unsupervised_networks[0] = init_values((int[]){784, 500, 784}, 3);
    unsupervised_networks[1] = init_values((int[]){500, 300, 500}, 3);

    connections_t *unsupervised_connections[unsupervised_layers];
    unsupervised_connections[0] = init_connections((int[]){784, 500, 784}, 3);
    unsupervised_connections[1] = init_connections((int[]){500, 300, 500}, 3);
    for(int i = 0; i < unsupervised_layers; i++) randomise_all_connections(unsupervised_connections[i], 10);

    double base_training_rate = strtod(argv[1], NULL);
    double reduction = strtod(argv[2], NULL);

    for(int supersession = 0; supersession < 40; supersession++){
        FILE *training_data = fopen("data/mnist_train.csv", "r");

        double training_rate = calc_trainingRate(base_training_rate, reduction, supersession);

        printf("##########\nsupersession: %d\ntrainingrate: %f\n##########\n", supersession, training_rate);

        char buffer[1<<13];
        //dups first line
        if(!fgets(buffer, 1<<13, training_data)) exit(EXIT_FAILURE);
        for(int i = 0; i < 50; i++){

            int num_of_layers_to_train = unsupervised_layers < (supersession + 1) ? unsupervised_layers : (supersession + 1);
            
            printf("\tsession: %d\n", i);

            int print = (i==0 || i==49) ? 1 : 0;
            training_session(unsupervised_networks, unsupervised_connections, training_data, training_rate, num_of_layers_to_train, print);
        }
        fclose(training_data);

        for(int i = 0; i < unsupervised_layers; i++){
            memcpy(unsup_tester_c->connections[i], unsupervised_connections[i]->connections[0], sizeof(double) * unsupervised_connections[i]->connection_sizes[0]);
            memcpy(unsup_tester_c->connections[(2*unsupervised_layers - i) - 1], unsupervised_connections[i]->connections[1], sizeof(double) * unsupervised_connections[i]->connection_sizes[1]);
        }

        set_first_layer(unsup_tester_n, unsupervised_networks[0]->layers[0]);
        apply_connections(unsup_tester_n, unsup_tester_c);

        printf("\toutput of full stack: 0,");
        for(int i = 0; i < unsup_tester_n->layer_sizes[unsup_tester_n->layer_sizes_length - 1]; i++){
            int output = (int)(unsup_tester_n->layers[unsup_tester_n->layer_sizes_length - 1][i] * 256);
            printf("%d,", output);
        }
        printf("\n");

        char outname[64];
        snprintf(outname, 64, "%s%d.212", argv[3], supersession);
        save_in_file(unsup_tester_n, unsup_tester_c, outname);
    }

    return 0;
}