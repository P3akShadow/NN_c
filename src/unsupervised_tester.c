#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "network.h"
#include "network_trainer.h"


int verbose = 0;
void fill_labels(long labels[], double content[], int input_length, FILE* source){

    for(int i = 0; i < 1000; i++){
        if(!fread(labels + i, sizeof(long), 1, source)) exit(EXIT_FAILURE);
        if(!fread(content + (i*input_length), sizeof(double) * input_length, 1, source)) exit(EXIT_FAILURE);
    }
}

void write_out(long labels[], double content[], values_t* new_net, connections_t* new_con, FILE* dump_data){
    for(int j = 0; j < 1000; j ++){
        set_first_layer(new_net, content + j * new_net->layer_sizes[0]);
        apply_connections(new_net, new_con);

        double* out = new_net->layers[new_net->layer_sizes_length - 1];

        fprintf(dump_data, "%ld,", labels[j]);
        
        for(int i = 0; i < new_net->layer_sizes[new_net->layer_sizes_length - 1]; i++){
            fprintf(dump_data, "%d,", (int) (256 * out[i]));
        }
        fprintf(dump_data, "\n");
    }
}

int main (int argc, char* argv[]){
    values_t* temp_net = NULL;
    connections_t* temp_con = NULL;

    int num_of_unsup = 2;

    recover_network_form_file(&temp_net, &temp_con, argv[1]);

    values_t* new_net = init_values(temp_net->layer_sizes + num_of_unsup, (temp_net->layer_sizes_length) - num_of_unsup);
    connections_t* new_con = init_connections(temp_net->layer_sizes + num_of_unsup, (temp_net->layer_sizes_length) - num_of_unsup);

    for(int i = 0; i < new_con->connection_sizes_length; i++){
        memcpy(new_con->connections[i], temp_con->connections[i + num_of_unsup], sizeof(double)*new_con->connection_sizes[i]);
    }

    FILE *test_data = fopen("data/unsup/bin_mnist_test_300", "r");
    FILE *dump_data = fopen("saves/unsup/unsup_apply_validate_test_300", "w");


    long labels[1000];
    double content[1000 * new_net->layer_sizes[0]];

    for(int i = 0; i < 60; i++){
        printf("printin %d\n", i);
        fill_labels(labels, content, new_net->layer_sizes[0] ,test_data);

        write_out(labels, content, new_net, new_con, dump_data);
    }

    fclose(test_data);
    fclose(dump_data);
}