#define MAX_LAYERS 8

typedef struct values {
    int layer_sizes_length;
    int layer_sizes[MAX_LAYERS];
    double *layers[MAX_LAYERS];
} values_t;

//connection from neuron <i, j1> to <i+1, j2> at i*(layers[i]+1)+j
//+1 because bias is saved together with weights
typedef struct connections {
    int connection_sizes_length;
    int connection_sizes[MAX_LAYERS-1];
    double *connections[MAX_LAYERS-1];
} connections_t;

values_t* init_values(int layer_sizes[], int layer_sizes_length);
connections_t* init_connections(int layer_sizes[], int layer_sizes_length);

void set_first_layer(values_t *network, double *first_layer);
double* get_layer(values_t *network, int layer);

void randomise_all_connections(connections_t *connections, double max_value);
void apply_connections(values_t* network, connections_t* connections);

void save_in_file(values_t* network, connections_t* connections, const char* filename);
void recover_network_form_file(values_t** network, connections_t** connections, const char* filename);