#include "NN_configure.h"
#include "NN_types.h"
#include <stdlib.h>

extern char s_str_error_string[80];

static double **s_g_NN_neurons = NULL;
static double ***s_g_NN_weights = NULL;

/*
 * TODO: fix bug with mem
 */
NN_BOOL NN_build(NN_configure_t const * const p_configure_params)
{
    int i,j;



    // allocate mem for neurons
    s_g_NN_neurons = NULL;
    s_g_NN_neurons = malloc (sizeof(double*) * p_configure_params->layer_count);
    if (!s_g_NN_neurons)
    {
        return NN_FALSE;
    }

    for (i = 0; i < p_configure_params->layer_count; ++i)
    {
        s_g_NN_neurons[i] = NULL;
        if (p_configure_params->b_consist_bias)
            s_g_NN_neurons[i] = malloc (sizeof(double) * p_configure_params->neurons_count[i] + i);
        else
            s_g_NN_neurons[i] = malloc (sizeof(double) * p_configure_params->neurons_count[i] + i);
    }



    // allocate mem for weights
    s_g_NN_weights = NULL;
    s_g_NN_weights = malloc(sizeof(double**) * p_configure_params->layer_count-1);
    for (i = 0; i < p_configure_params->layer_count -1; ++i)
    {
        if (p_configure_params->b_consist_bias)
            s_g_NN_weights[i] = malloc(sizeof(double*) *p_configure_params->neurons_count[i] +1);
        else
            s_g_NN_weights[i] = malloc(sizeof(double*) *p_configure_params->neurons_count[i]);
        
        for (j = 0; j < p_configure_params->neurons_count[i]; ++j)
        {

            s_g_NN_weights[i][j] = malloc(sizeof(double) * p_configure_params->neurons_count[i+1]);
        }
    }
    return NN_TRUE;
}

static void initialize_weights_with( NN_configure_t const * const p_configure, double const * const p_weights, const size_t weight_count)
{
    int i,j,j2, w_count = 0;
    for (i = 0; i < (p_configure->layer_count - 1) && (w_count < weight_count); ++i)
    {
        for (j = 0; j < (p_configure->neurons_count[i] + (p_configure->b_consist_bias != 0) ? 1:0) && (w_count < weight_count); ++j)
        {
            for (j2 = 0; j2 < p_configure->neurons_count[i+1] && (w_count < weight_count); ++j2)
            {   
                s_g_NN_weights[i][j][j2] = p_weights[w_count];
                ++w_count;
            }
        }
    }
}

void push_values_into_NN(NN_configure_t const * const p_configure, double const * const p_values, const size_t value_count)
{
    int i;
    for (i = 0; (i < value_count) && (i < p_configure->neurons_count[0]); ++i)
    {
        s_g_NN_neurons[0][i] = p_values[i];
    }
}