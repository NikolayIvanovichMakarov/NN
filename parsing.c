#include "parsing.h"

static const char s_str_error_string[80] = {""};

typedef struct NN_loading_params_s
{
    int layer_count;    // count of layers;
    int neurons_count[MAX_LAYERS_COUNT];
    int b_consist_bias;
} NN_loading_params_t;

#define NN_PARSING_TOO_LONG_LAYER_COUNT 1
#define NN_PARSING_NO_FILE_CONSIST 1
#define NN_PARSING_NOT_CORRECT_DATA_FORMAT 1
#define NN_NO_PROBLEM 1

typedef int NN_PARSING_ERROR_CODE ;

#define NN_TRUE 1
#define NN_FALSE 0

NN_PARSING_ERROR_CODE parse(char const * const str_filename, NN_loading_params_t * const p_loading_params)
{
    #define ERR_EXIT(error_str,p_file, err_code) p_loading_params->layer_count = 0, sprintf(s_str_error_string, STR), fclose(p_file), err_code;
    #define ERR_EXIT_FORMAT(p_file,err_code,error_str, ...) p_loading_params->layer_count = 0, sprintf(s_str_error_string, error_str, ##__VA_ARGS__), fclose(p_file), err_code
    #define CLEAN_NN_LOADING_PARAMS(p_lp) memset(p_lp, 0, sizeof(NN_loading_params_t));
    
    int i;
    FILE *p_file_in = fopen(str_filename, "r");
    
    if (p_file_in == NULL)
    {
        return ERR_EXIT_FORMAT(p_file_in, NN_PARSING_NO_FILE_CONSIST, "file %s is not consist", str_filename);
    }

    if (fscanf(p_file_in,"%d", &p_loading_params->layer_count) == 0)
    {
        return ERR_EXIT_FORMAT(p_file_in, NN_PARSING_NOT_CORRECT_DATA_FORMAT, "Not correct file format: layer count is broken");
    }

    if (p_loading_params->layer_count > MAX_LAYERS_COUNT)
    {
        return ERR_EXIT_FORMAT(p_file_in, NN_PARSING_TOO_LONG_LAYER_COUNT, "layer count %d is larger than max layer count %d", p_loading_params->layer_count, MAX_LAYERS_COUNT);
    }
    
    if (p_loading_params->layer_count < MIN_LAYERS_COUNT)
    {
        return ERR_EXIT_FORMAT(p_file_in, NN_PARSING_TOO_LONG_LAYER_COUNT, "layer count %d is less then min layer count %d", p_loading_params->layer_count, MIN_LAYERS_COUNT);
    }

    if (fscanf(p_file_in,"%d", &p_loading_params->b_consist_bias) == 0)
    {
        return ERR_EXIT_FORMAT(p_file_in, NN_PARSING_NOT_CORRECT_DATA_FORMAT, "Not correct file format: bias is broken");
    }
    


    for (i = 0; i < p_loading_params->layer_count; ++i)
    {
        if ( fscanf(p_file_in,"%d", &p_loading_params->neurons_count[i]) == 0)
        {
            return ERR_EXIT_FORMAT(p_file_in, NN_PARSING_NOT_CORRECT_DATA_FORMAT, "Not correct file format: neuron count at layer %d is broken", (i+1));
        }
        else if (p_loading_params->neurons_count[i] > MAX_NEURONS_COUNT)
        {
            return ERR_EXIT_FORMAT(p_file_in, NN_PARSING_TOO_LONG_LAYER_COUNT, "neuron count %d at layer %d is larger than max neuron count %d", p_loading_params->neurons_count[i], (i+1), MAX_NEURONS_COUNT);
        }
        else if (p_loading_params->neurons_count[i] < MIN_NEURONS_COUNT)
        {
            return ERR_EXIT_FORMAT(p_file_in, NN_PARSING_TOO_LONG_LAYER_COUNT, "neuron count %d at layer %d is larger less then min neuron count %d", p_loading_params->neurons_count[i], (i+1), MIN_NEURONS_COUNT);
        }
    }

    return NN_NO_PROBLEM;
};
