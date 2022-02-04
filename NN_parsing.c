#include "NN_types.h"
#include "NN_parsing.h"
#include "NN_configure.h"
#include "NN_learn_dataset.h"

static char s_str_error_string[80] = "No error";
static char *s_str_buf_error_string[80];

char* NN_get_parse_error()
{
    strcpy(s_str_buf_error_string, s_str_error_string);
    return s_str_buf_error_string;
}


NN_BOOL NN_parse(char const * const str_filename, NN_configure_t * const p_loading_params)
{
    int i;
    FILE *p_file_in;

    #define ERR_EXIT(error_str,p_file, err_code) p_loading_params->layer_count = 0, sprintf(s_str_error_string, STR), fclose(p_file), err_code;
    #define ERR_EXIT_FORMAT(p_file,err_code,error_str, ...) sprintf(s_str_error_string, error_str, ##__VA_ARGS__), p_loading_params->layer_count = 0, fclose(p_file), err_code



    p_file_in = fopen(str_filename, "r");
    if (!p_file_in)
    {
        return NN_FALSE;
    }

    if (fscanf(p_file_in,"%d", &p_loading_params->layer_count) == 0)
    {
        return ERR_EXIT_FORMAT(p_file_in, /*NN_PARSING_NOT_CORRECT_DATA_FORMAT*/ NN_FALSE, "Not correct file format: layer count is broken");
    }

    if (p_loading_params->layer_count > MAX_LAYERS_COUNT)
    {
        return ERR_EXIT_FORMAT(p_file_in, /*NN_PARSING_TOO_LONG_LAYER_COUNT*/ NN_FALSE, "layer count %d is larger than max layer count %d", p_loading_params->layer_count, MAX_LAYERS_COUNT);
    }
    
    if (p_loading_params->layer_count < MIN_LAYERS_COUNT)
    {
        return ERR_EXIT_FORMAT(p_file_in, /*NN_PARSING_TOO_LONG_LAYER_COUNT*/ NN_FALSE, "layer count %d is less then min layer count %d", p_loading_params->layer_count, MIN_LAYERS_COUNT);
    }

    if (fscanf(p_file_in,"%d", &p_loading_params->b_consist_bias) == 0)
    {
        return ERR_EXIT_FORMAT(p_file_in, /*NN_PARSING_NOT_CORRECT_DATA_FORMAT*/ NN_FALSE, "Not correct file format: bias is broken");
    }

    for (i = 0; i < p_loading_params->layer_count; ++i)
    {
        if ( fscanf(p_file_in,"%d", &p_loading_params->neurons_count[i]) == 0)
        {
            return ERR_EXIT_FORMAT(p_file_in, /*NN_PARSING_NOT_CORRECT_DATA_FORMAT*/ NN_FALSE, "Not correct file format: neuron count at layer %d is broken", (i+1));
        }
        else if (p_loading_params->neurons_count[i] > MAX_NEURONS_COUNT)
        {
            return ERR_EXIT_FORMAT(p_file_in, /*NN_PARSING_TOO_LONG_LAYER_COUNT*/ NN_FALSE, "neuron count %d at layer %d is larger than max neuron count %d", p_loading_params->neurons_count[i], (i+1), MAX_NEURONS_COUNT);
        }
        else if (p_loading_params->neurons_count[i] < MIN_NEURONS_COUNT)
        {
            return ERR_EXIT_FORMAT(p_file_in, /*NN_PARSING_TOO_LONG_LAYER_COUNT*/ NN_FALSE, "neuron count %d at layer %d is larger less then min neuron count %d", p_loading_params->neurons_count[i], (i+1), MIN_NEURONS_COUNT);
        }
    }

    return NN_TRUE;

    #undef ERR_EXIT
    #undef ERR_EXIT_FORMAT
}

static NN_BOOL s_NN_alloc_line(NN_configure_t  const * const p_configure_params, NN_learn_line_t * const p_learn_line)
{
    p_learn_line->p_inputs = NULL;
    p_learn_line->p_inputs = malloc(sizeof(double) * p_configure_params->neurons_count[0]);
    if (p_learn_line == NULL)
        return NN_FALSE;

    p_learn_line->p_targets = NULL;
    p_learn_line->p_targets = malloc(sizeof(double) * p_configure_params->neurons_count[p_configure_params->layer_count-1]);
    if (p_learn_line->p_targets == NULL)
    {
        free(p_learn_line->p_targets);
        return NN_FALSE;
    }
    
    return NN_TRUE;
}

static NN_BOOL s_NN_learn_line_parse(FILE * const p_file, NN_configure_t  const * const p_configure_params, NN_learn_line_t * const p_learn_line)
{
    int test_line_i;
    if (p_configure_params == NULL)
        return NN_FALSE;

    for (test_line_i = 0; test_line_i < p_configure_params->neurons_count[0]; ++test_line_i)
    {
        if (fscanf(p_file, "%lf",&p_learn_line->p_inputs[test_line_i]) == 0)
            return NN_FALSE;
    }

    for (test_line_i = 0; test_line_i < p_configure_params->neurons_count[p_configure_params->layer_count-1]; ++test_line_i)
    {
        if (fscanf(p_file, "%lf",&p_learn_line->p_targets[test_line_i]) == 0)
            return NN_FALSE;
    }

    return NN_TRUE;
}

NN_BOOL NN_learn_dataset_parse(char const * const str_filename, NN_configure_t const * const p_loading_params, NN_learn_dataset_t * const p_learn_dataset )
{
    FILE *p_file_in;
    p_file_in = fopen(str_filename, "r");
    int line_count = 0;
    int i;
    if (p_file_in == NULL)
    {
        return NN_FALSE;
    }

    if (p_learn_dataset == NULL)
    {
        fclose(p_file_in);
        return NN_FALSE;
    }

    fscanf(p_file_in,"%d",&p_learn_dataset->total_learn_line_count);

    if (p_learn_dataset->total_learn_line_count <= 0)
    {
        fclose(p_file_in);
        return NN_FALSE;
    }

    p_learn_dataset->p_learn_lines = malloc(sizeof(NN_learn_line_t)* p_learn_dataset->total_learn_line_count);

    for (int i = 0; i < p_learn_dataset->total_learn_line_count; ++i)
    {
        if (s_NN_alloc_line(p_loading_params, &p_learn_dataset->p_learn_lines[i]) == NN_FALSE)
        {
            fclose(p_file_in);
            return NN_FALSE;
        }
        if (s_NN_learn_line_parse(p_file_in, p_loading_params, &p_learn_dataset->p_learn_lines[i]) == NN_FALSE)
        {
            fclose(p_file_in);
            return NN_FALSE;
        }
    }
    
    return NN_TRUE;
}

int NN_weights_parse(char const * const str_filename,double * const p_weights)
{
    FILE *p_file_in = NULL;
    int weights_count = 0;


    p_file_in = fopen(str_filename, "r");
    if (p_file_in == NULL)
        return 0;
    while (fscanf(p_file_in, "%lf", &p_weights[weights_count]) == 1)
    {
        ++weights_count;
    }
    
    fclose(p_file_in);
    return weights_count;
}