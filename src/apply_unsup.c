#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "network.h"
#include "network_trainer.h"


int verbose = 0;
void fill_labels(long labels[], double content[], FILE* source){
    char buffer[1<<13];

    for(int i = 0; i < 1000; i++){
        if(!fgets(buffer, 1<<13, source)) exit(EXIT_FAILURE);
        
        labels[i] = strtol(strtok(buffer, ","), NULL, 10);
        
        for(int j = 0; j < 28*28; j++){
            long intermediate = strtol(strtok(NULL, ","), NULL, 10);
            content[i*28*28+j] = ((double) intermediate) / 256;
        }
    }
}

void dump_part(long labels[1000], double content[1000 * 28*28], values_t* new_net, connections_t* new_con, FILE* dump_data){
    for(int j = 0; j < 1000; j ++){
        set_first_layer(new_net, content + j * 28 * 28);
        apply_connections(new_net, new_con);

        double* out = new_net->layers[new_net->layer_sizes_length - 1];
        fwrite(labels + j, sizeof(long), 1, dump_data);
        fwrite(out, sizeof(double) * (new_net->layer_sizes[new_net->layer_sizes_length - 1]), 1, dump_data);
    }
}

int main (int argc, char* argv[]){
    values_t* new_net = NULL;
    connections_t* new_con = NULL;

    int num_of_unsup = 2;

    recover_network_form_file(&new_net, &new_con, argv[1]);

    new_net->layer_sizes_length = num_of_unsup + 1;
    new_con->connection_sizes_length = num_of_unsup;

    FILE *test_data = fopen("data/mnist_test.csv", "r");
    FILE *dump_data = fopen("data/unsup/bin_mnist_test_300", "w");


    long labels[1000];
    double content[1000 * 28*28];

    char dump[1<<13];
    if(!fgets(dump, 1<<13, test_data)) exit(EXIT_FAILURE);

    for(int i = 0; i < 60; i++){
        printf("printin %d\n", i);
        fill_labels(labels, content, test_data);

        dump_part(labels, content, new_net, new_con, dump_data);
    }

    fclose(test_data);
    fclose(dump_data);
}