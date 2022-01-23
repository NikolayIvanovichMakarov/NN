#include "NN_types.h"
#include "NN_parsing.h"
#include "NN_configure.h"

extern char s_str_error_string[80];

NN_PARSING_ERROR_CODE parse(char const * const str_filename, NN_configure_t * const p_loading_params)
{
    int i;
    FILE *p_file_in;

    #define ERR_EXIT(error_str,p_file, err_code) p_loading_params->layer_count = 0, sprintf(s_str_error_string, STR), fclose(p_file), err_code;
    #define ERR_EXIT_FORMAT(p_file,err_code,error_str, ...) sprintf(s_str_error_string, error_str, ##__VA_ARGS__), p_loading_params->layer_count = 0, fclose(p_file), err_code



    p_file_in = fopen(str_filename, "r");
    if (!p_file_in)
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