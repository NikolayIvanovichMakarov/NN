#define MAX_LAYERS_COUNT 100

typedef struct NN_configure_s
{
    int layer_count;    // count of layers;
    int neurons_count[MAX_LAYERS_COUNT];
    int b_consist_bias;
} NN_configure_t;

void print_configure_params(NN_configure_t const * const p_loading_params);
void save_configure_params_into_file(char const *const str_filename, NN_configure_t const * const p_loading_params);
