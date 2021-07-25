#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "network.h"
#include "network_trainer.h"

int verbose = 0;

void fill_labels(long labels[1000], double content[1000 * 28*28], FILE* source){
    char buffer[1<<13];

    for(int i = 0; i < 1000; i++){
        fgets(buffer, 1<<13, source);
        labels[i] = strtol(strtok(buffer, ","), NULL, 10);

        for(int j = 0; j < 28*28; j++){
            long intermediate = strtol(strtok(NULL, ","), NULL, 10);
            content[i*28*28+j] = ((double) intermediate) / 256;
        }
    }
}

int check_part(long labels[1000], double content[1000 * 28*28], values_t* new_net, connections_t* new_con){
    int correct = 0;
    
    for(int j = 0; j < 1000; j ++){
        set_first_layer(new_net, content + j * 28 * 28);
        apply_connections(new_net, new_con);


        double* out = get_layer(new_net, new_net->layer_sizes_length - 1);
        int max_index = 0;
        for(int i = 0; i < new_net->layer_sizes[new_net->layer_sizes_length - 1]; i++){
            if(out[max_index] < out[i]) max_index = i;
        }

        if(max_index == labels[j]){
            //printf("SUCCESS! %d ", max_index);
            correct++;
        }
        else if(verbose) {
            printf("ERROR expected %ld but got %d\n", labels[j], max_index);
        }
    }
    return correct;
}

int main (int argc, char* argv[]){
    values_t* new_net = NULL;
    connections_t* new_con = NULL;

    recover_network_form_file(&new_net, &new_con, argv[1]);

    FILE *test_data = fopen("data/mnist_test.csv", "r");

    long labels[1000];
    double content[1000 * 28*28];

    int correct = 0;

    char dump[1<<13];
    fgets(dump, 1<<13, test_data);

    for(int i = 0; i < 10; i++){
        fill_labels(labels, content, test_data);

        correct += check_part(labels, content, new_net, new_con);
    }

    printf("num of correct guesses %d\n", correct);

}