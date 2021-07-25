#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "network.h"
#include "cblas.h"

#define ADJUST_LAYER_SIZE 1

/**
 * @brief Initializes a struct for the values of a neuronal Network 
 * @param layer_sizes Sizes of the Layers, Maximum length = MAX_LAYERS
 * @return space for the values of a neuronal network
 */
values_t* init_values(int layer_sizes[], int layer_sizes_length){
    int sum_of_values = 0;

    for(int i = 0; i < layer_sizes_length; i++){
        sum_of_values += layer_sizes[i];
    }

    void* value_space = malloc(sizeof(values_t) + sizeof(double) * sum_of_values);
    memset(value_space, 0, sizeof(values_t) + sizeof(double) * sum_of_values);

    values_t* values = value_space;
    values->layer_sizes_length = layer_sizes_length;
    for(int i = 0; i < layer_sizes_length; i++){
        values->layer_sizes[i] = layer_sizes[i];
    }

    sum_of_values = 0;
    double* start_of_layers = (double*) ((u_int8_t*) value_space + sizeof(values_t));

    for(int i = 0; i < layer_sizes_length; i++){
        values->layers[i] = start_of_layers + sum_of_values;
        sum_of_values += layer_sizes[i];
    }

    return values;
}

/**
 * @brief Initializes a struct for the connections of a neuronal Network 
 * @param layer_sizes Sizes of the Layers, Maximum length = MAX_LAYERS
 * @return space for the connections of a neuronal Netowork
 */
connections_t* init_connections(int layer_sizes[], int layer_sizes_length){
    int sum_of_weights = 0;

    for(int i = 0; i < layer_sizes_length - 1; i++){
        // +1 because the last weight will be the bias
        sum_of_weights += (layer_sizes[i] + 1) * layer_sizes[i+1] ;
    }

    void* connection_space = malloc(sizeof(connections_t) + sizeof(double) * sum_of_weights);
    memset(connection_space, 0, sizeof(connections_t) + sizeof(double) * sum_of_weights);

    connections_t* connections = connection_space;
    connections->connection_sizes_length = layer_sizes_length - 1;
    for(int i = 0; i < layer_sizes_length - 1; i++){
        connections->connection_sizes[i] = (layer_sizes[i] + 1) * layer_sizes[i+1];
    }

    sum_of_weights = 0;
    double* start_of_connections = (double*) ((u_int8_t*) connection_space + sizeof(connections_t));

    for(int i = 0; i < layer_sizes_length - 1; i++){
        connections->connections[i] = start_of_connections + sum_of_weights;
        sum_of_weights += (layer_sizes[i] + 1) * layer_sizes[i+1];
    }

    for(int i = 0; i < 100; i++){}

    return connections;
}

/**
 * @brief Applies the sigmoid function to the input and returns the value
 * @param x input value
 * @return the value of sigmoid(x)
 */
static inline double sigmoid(double x) {
    return 1/(1+pow(M_E, (-x)));
}

/**
 * @brief Sets the first layer of the network to the given input layer
 * @param network to set the first layer
 * @param first_layer the values to set, must have a size of network->connection_sizes[0]
 */
void set_first_layer(values_t *network, double *first_layer){
    memcpy(network->layers[0], first_layer, network->layer_sizes[0] * sizeof(double));
}

/**
 * @brief Gets a layer
 * @param network to get a layer from
 * @param layer index of layer
 * @return pointer to the output
 */
double* get_layer(values_t *network, int layer){
    return(network->layers[layer]);
}

/**
 * @brief Gets a value of the network
 * @param network to get value from
 * @param layer line of the value
 * @param index index of the value
 * @return the value
 */
double get_value(values_t *network, int layer, int index){
    return(*(network->layers[layer] + index));
}

/**
 * @brief Gets the full co
 * nnection to a value
 * @param network the network the value is in
 * @param connections to get a connection from
 * @param layer index of layer that the value is in
 * @param index index of the value has
 * @return pointer to the connection
 */
double* get_connection_to(values_t *network, connections_t *connections, int layer, int index){
    int index_offset = network->layer_sizes[layer - 1] + 1;
    return(connections->connections[layer-1] + index * index_offset);
}

/**
 * @brief Sets the values of a connection
 * @param network the network the value is in
 * @param connections to set a connection in
 * @param layer index of layer that the value is in
 * @param index index of the value has
 * @param weights pointer to values of new weights (must have appropriate length)
 * @return pointer to the connection
 */
void set_connection_to(values_t *network, connections_t *connections, int layer, int index, double* weights){
    int index_offset = network->layer_sizes[layer - 1] + 1;
    memcpy(connections->connections[layer-1] + index * index_offset, weights, index_offset * sizeof(double));
}

void randomise_all_connections(connections_t *connections, double max_value){
    srand(2);
    for(int i = 0; i < connections->connection_sizes_length; i++){
        for(int j = 0; j < connections->connection_sizes[i]; j++){
            *(connections->connections[i] + j) = max_value - 2 * max_value * ((double)rand()/RAND_MAX);
        }
    }
}

/**
 * @brief Applies the given connections to the given network
 * @param network the network with the first layer set, the others will be changed
 * @param connections the connections to calculate
 */
void apply_connections(values_t* network, connections_t* connections){
    for(int i = 0; i < connections->connection_sizes_length; i++){
        int input_size = network->layer_sizes[i];
        int result_size = network->layer_sizes[i+1];

        double *input = network->layers[i];
        double *matrix = connections->connections[i];
        double *result = network->layers[i+1];

        memset(result, 0, sizeof(double)*result_size);

        cblas_dgemv(CblasRowMajor, CblasNoTrans, result_size, input_size, 1.0, matrix, input_size + 1, input, 1, 1.0, result, 1);

        for(int j = 0; j < network->layer_sizes[i+1]; j++) {
            //bias
            result[j] += (*(connections->connections[i] + (network->layer_sizes[i] + j * (network->layer_sizes[i] + 1))));

            if(ADJUST_LAYER_SIZE){
                *(network->layers[i + 1] + j) = sigmoid(result[j]/network->layer_sizes[i]);
            } else {
                *(network->layers[i + 1] + j) = sigmoid(result[j]);
            }
        }
    }
}

void save_in_file(values_t* network, connections_t* connections, const char* filename){
    FILE* output = fopen(filename, "w");

    fwrite(network, sizeof(values_t), 1, output);
    for(int i = 0; i < network->layer_sizes_length; i++){
        fwrite(network->layers[i], sizeof(double) * network->layer_sizes[i], 1, output);
    }
    fwrite(connections, sizeof(connections_t), 1, output);
    for(int i = 0; i < connections->connection_sizes_length; i++){
        fwrite(connections->connections[i], sizeof(double) * connections->connection_sizes[i], 1, output);
    }

    fflush(output);
    fclose(output);
}

void recover_network_form_file(values_t** network_place, connections_t** connections_place, const char* filename){
    FILE* input = fopen(filename, "r");

    u_int8_t throwaway[1024];
    int needed_buff_size = 0;
    int read_bytes = 0;

    do{
        read_bytes = fread(throwaway, 1, 1024, input);
        needed_buff_size += read_bytes;
    } while(read_bytes != 0);
    fclose(input);

    input = fopen(filename, "r");
    void* space = malloc(needed_buff_size);
    memset(space, 0, needed_buff_size);
    fread(space, 1, needed_buff_size, input);
    fclose(input);

    values_t* network = space;
    double* start_of_layers = (double*) ((u_int8_t*) space + sizeof(values_t));
    int sum_of_values = 0;
    for(int i = 0; i < network->layer_sizes_length; i++){
        network->layers[i] = start_of_layers + sum_of_values;
        sum_of_values += network->layer_sizes[i];
    }

    connections_t* connections = (connections_t*) (start_of_layers + sum_of_values);
    int sum_of_weights = 0;
    double* start_of_connections = (double*) ((u_int8_t*) connections + sizeof(connections_t));

    for(int i = 0; i < connections->connection_sizes_length; i++){
        connections->connections[i] = start_of_connections + sum_of_weights;
        sum_of_weights += connections->connection_sizes[i];
    }

    (*network_place) = network;
    (*connections_place) = connections;
    
}

void test () {
    values_t *network = init_values((int[]){2, 2, 1}, 3);
    connections_t *connections = init_connections((int[]){2, 2, 1}, 3);

    randomise_all_connections(connections, 2);
    set_first_layer(network, (double[]){1, 0.5});

    apply_connections(network, connections);

    for(int i = 0; i < network->layer_sizes_length; i++){
        for(int j = 0; j < network->layer_sizes[i]; j++){
            printf("the value of neuron on position <%d, %d> is: %f\n", i, j, *(network->layers[i] + j));
        }
    }
}