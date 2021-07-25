#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "network.h"
#include "cblas.h"

void calc_particular_deltas(values_t* network, connections_t* connections, double* expected_output, values_t* delta_save, int print_deltas){
    int last_layer_index = delta_save->layer_sizes_length - 1;

    double sum = 0;

    for(int i = 0; i < delta_save->layer_sizes[last_layer_index]; i++){
        double current_value = *(network->layers[last_layer_index] + i);
        
        double first_error = (expected_output[i] - current_value);

        //add some hard limit to the error in order to not let it tank too hard if tiny
        if(first_error > 0) first_error += 0.3;
        if(first_error < 0) first_error -= 0.3;
        if(first_error > 0.35) first_error += 0.4;
        if(first_error < -0.35) first_error -= 0.4;

        //(sometimes) raise to a power to emphasize big diffs
        first_error = first_error * first_error * first_error * first_error * first_error;

        if(first_error > 10) first_error = 10;
        if(first_error < -10) first_error = -10;


        *(delta_save->layers[last_layer_index] + i) = (current_value * (1 -current_value) * first_error);

        sum += fabs(*(delta_save->layers[last_layer_index] + i));
    }

    if(print_deltas) printf("\t\tlayer %d has an average of %f\n", last_layer_index, sum / delta_save->layer_sizes[last_layer_index]);


    for(int layer = delta_save->layer_sizes_length - 2; layer > 0; layer--){
        memset(delta_save->layers[layer], 0, sizeof(double)*delta_save->layer_sizes[layer]);

        cblas_dgemv(CblasRowMajor, CblasTrans, delta_save->layer_sizes[layer + 1], delta_save->layer_sizes[layer], 1.0, connections->connections[layer], delta_save->layer_sizes[layer] + 1,
        delta_save->layers[layer + 1], 1, 1.0, delta_save->layers[layer], 1);

        double previous_layer_root = sqrt((double) delta_save->layer_sizes[layer + 1]);

        sum = 0; 

        for(int neuron = 0; neuron < delta_save->layer_sizes[layer]; neuron++){
            double current_value = *(network->layers[layer] + neuron);

            //adjust for gradient
            *(delta_save->layers[layer] + neuron) *= current_value * (1 - current_value);

            //adjust for the previous layer size (root because I think of that as a sum of random values that have an expected value of 0)
            *(delta_save->layers[layer] + neuron) /= previous_layer_root;

            sum += fabs(*(delta_save->layers[layer] + neuron));
        }

        if(print_deltas) printf("\t\tlayer %d has an average of %f\n", layer, sum / delta_save->layer_sizes[layer]);
    }
}

/*
void gradient_decent(values_t* network, connections_t* connections, double* input, double* expected_output, double mult_factor){
    set_first_layer(network, input);
    apply_connections(network, connections);

    values_t* particular_deltas = init_values(network->layer_sizes, network->layer_sizes_length);
    calc_particular_deltas(network, connections, expected_output, particular_deltas);

    for(int layer_index = network->layer_sizes_length - 1; layer_index > 0 ; layer_index--){

        int next_layer_size = network->layer_sizes[layer_index];
        int prevoious_layer_size = network->layer_sizes[layer_index - 1];

        double* connections_to_prev = connections->connections[layer_index - 1];

        for(int neuron_index = 0; neuron_index < next_layer_size; neuron_index++){
            double delta = *(particular_deltas->layers[layer_index] + neuron_index);

            for(int previous_neuron_index = 0; previous_neuron_index < prevoious_layer_size; previous_neuron_index++){
                double previous_neuron_value = *(network->layers[layer_index - 1] + previous_neuron_index);
                double* weight = connections_to_prev + neuron_index * (prevoious_layer_size + 1) + previous_neuron_index;

                *weight += delta * previous_neuron_value * mult_factor;
            }

            //bias
            double* weight = connections_to_prev + neuron_index * (prevoious_layer_size + 1) + prevoious_layer_size;
            *weight += delta * 1 * mult_factor;
        }
    }

    free(particular_deltas);
}
*/

/*
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
##############################################
-----------DONT DELETE blas version-----------
##############################################
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

This is the openblas version
it seems to be slower on my machine but god knows how it scales
*/
void gradient_decent(values_t* network, connections_t* connections, double* input, double* expected_output, double mult_factor, int print_deltas){
    set_first_layer(network, input);
    apply_connections(network, connections);

    values_t* particular_deltas = init_values(network->layer_sizes, network->layer_sizes_length);
    calc_particular_deltas(network, connections, expected_output, particular_deltas, print_deltas);

    for(int layer_index = network->layer_sizes_length - 1; layer_index > 0 ; layer_index--){

        int next_layer_size = particular_deltas->layer_sizes[layer_index];
        int previous_layer_size = network->layer_sizes[layer_index - 1];

        double *errors = particular_deltas->layers[layer_index];
        double prev_values[previous_layer_size + 1];

        
        memcpy(prev_values, network->layers[layer_index - 1], sizeof(double)*previous_layer_size);
        
        //weight
        prev_values[previous_layer_size] = 1;

        double* connections_to_prev = connections->connections[layer_index - 1];

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, next_layer_size, previous_layer_size + 1, 1, mult_factor, errors, 1, prev_values, previous_layer_size + 1, 1.0, connections_to_prev, previous_layer_size + 1);
    }

    free(particular_deltas);
}

//performs bad
void gradient_decent_bulk(values_t* network, connections_t* connections, int bulksize, double* inputs, double* expected_outputs, double mult_factor){

    values_t* sum_deltas = init_values(network->layer_sizes, network->layer_sizes_length);
    for(int example = 0; example < bulksize; example++){
        set_first_layer(network, inputs + (example * network->layer_sizes[0]));
        apply_connections(network, connections);
        double *expected_output = expected_outputs + (example * network->layer_sizes[network->layer_sizes_length - 1]);

        

        values_t* particular_deltas = init_values(network->layer_sizes, network->layer_sizes_length);
        calc_particular_deltas(network, connections, expected_output, particular_deltas, 0);

        //add errors of different values
        for(int layer_i = 1; layer_i < sum_deltas->layer_sizes_length; layer_i++){
            for(int neuron_i = 0; neuron_i < sum_deltas->layer_sizes[layer_i]; neuron_i++){
                sum_deltas->layers[layer_i][neuron_i] += particular_deltas->layers[layer_i][neuron_i];
            }
        }

        free(particular_deltas);
    }

    for(int layer_index = network->layer_sizes_length - 1; layer_index > 0 ; layer_index--){

        int next_layer_size = sum_deltas->layer_sizes[layer_index];
        int previous_layer_size = network->layer_sizes[layer_index - 1];

        double *errors = sum_deltas->layers[layer_index];
        double prev_values[previous_layer_size + 1];

        
        memcpy(prev_values, network->layers[layer_index - 1], sizeof(double)*previous_layer_size);
        
        //weight
        prev_values[previous_layer_size] = 1;

        double* connections_to_prev = connections->connections[layer_index - 1];

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, next_layer_size, previous_layer_size + 1, 1, mult_factor, errors, 1, prev_values, previous_layer_size + 1, 1.0, connections_to_prev, previous_layer_size + 1);
    }

    free(sum_deltas);
}