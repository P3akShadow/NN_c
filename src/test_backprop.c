#include <stdlib.h>
#include <stdio.h>

#include "network.h"
#include "network_trainer.h"

int main(){
    values_t *network = init_values((int[]){2, 2, 1}, 3);
    connections_t *connections = init_connections((int[]){2, 2, 1}, 3);

    *(connections->connections[0] + 0) = -0.15;
    *(connections->connections[0] + 1) = 0.64;
    *(connections->connections[0] + 3) = 0.90;
    *(connections->connections[0] + 4) = 0.08;

    *(connections->connections[1] + 0) = 0.24;
    *(connections->connections[1] + 1) = 0.98;

    set_first_layer(network, (double[]){0, 0});

    apply_connections(network, connections);

    for(int i = 0; i < network->layer_sizes_length; i++){
        for(int j = 0; j < network->layer_sizes[i]; j++){
            printf("layer: %d; neuron: %d; value %f;\n", i, j, *(network->layers[i] + j));
        }
    }

    gradient_decent(network, connections, (double[]){0, 0}, (double[]){0}, 1);

    apply_connections(network, connections);

    for(int i = 0; i < connections->connection_sizes_length; i++){
        for(int j = 0; j < connections->connection_sizes[i]; j++){
            printf("#WEIGHT layer: %d; weight: %d; value %f;\n", i, j, *(connections->connections[i] + j));
        }
    }

    for(int i = 0; i < network->layer_sizes_length; i++){
        for(int j = 0; j < network->layer_sizes[i]; j++){
            printf("#VALUE layer: %d; neuron: %d; value %f;\n", i, j, *(network->layers[i] + j));
        }
    }
}