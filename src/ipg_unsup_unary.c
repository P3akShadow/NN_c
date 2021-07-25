#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "network.h"
#include "network_trainer.h"

int verbose = 0;
int number_to_train = 0;

int fill_labels(long labels[], double content[], int input_length, FILE* source){
    int range_count = 0;
    for(int i = 0; i < 1000; i++){
        if(!fread(labels + i, sizeof(long), 1, source)) exit(EXIT_FAILURE);
        if(!fread(content + (i*input_length), sizeof(double) * input_length, 1, source)) exit(EXIT_FAILURE);

        range_count++;
    }

    return range_count;
}

int check_part(long labels[], double content[], int input_length, values_t* new_net, connections_t* new_con){
    int correct = 0;
    
    for(int j = 0; j < 1000; j ++){
        set_first_layer(new_net, content + j * input_length);
        apply_connections(new_net, new_con);

        //needed for partial output layer

        double* out = get_layer(new_net, new_net->layer_sizes_length - 1);
        int max_index = 0;

        for(int i = 0; i < new_net->layer_sizes[new_net->layer_sizes_length - 1]; i++){
            if(out[max_index] < out[i]) max_index = i;
        }

        //-x to make differnet pairs
        if((max_index == 0 && labels[j] == number_to_train) || (max_index == 1 && labels[j] != number_to_train)){
            //printf("SUCCESS! %d ", max_index);
            correct++;
        }
        else if(verbose){
            printf("ERROR at index %d: expected %d but another number seems more likely for the system\n", j, number_to_train);

            for(int i = 0; i < new_net->layer_sizes[new_net->layer_sizes_length - 1]; i++){
                printf("%d:%f; ", i, new_net->layers[new_net->layer_sizes_length - 1][i]);
            }
            printf("\n");
        }
    }
    return correct;
}

int main (int argc, char* argv[]){
    values_t* new_net = NULL;
    connections_t* new_con = NULL;

    if(argc > 3) verbose = strtol(argv[3], NULL, 10);

    recover_network_form_file(&new_net, &new_con, argv[2]);
    number_to_train = strtol(argv[1], NULL, 10);

    FILE *test_data = fopen("data/unsup/bin_mnist_test_300", "r");

    long labels[1000];
    int input_length = new_net->layer_sizes[0];
    double content[1000 * input_length];

    int range = 0;
    int correct = 0;

    for(int i = 0; i < 10; i++){
        if(verbose) printf("############\nbatch: %d\n############\n", i);
        range += fill_labels(labels, content, input_length, test_data);

        correct += check_part(labels, content, input_length, new_net, new_con);
    }

    printf("num of tries:           %d\n", range);
    printf("num of correct guesses: %d\n", correct);
    printf("relative:               %f\n", ((double)correct)/range);

}