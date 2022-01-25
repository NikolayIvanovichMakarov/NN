#include "NN_configure.h"
#include "NN_types.h"
#include <stdlib.h>

extern char s_str_error_string[80];

static double **s_g_NN_neurons = NULL;
static double ***s_g_NN_weights = NULL;
static double **s_g_NN_errors = NULL;

#define LAST_LAYER(p_configure) (p_configure->layer_count-1)
#define PRE_LAST_LAYER(p_configure) (p_configure->layer_count-2)
#define NEURON_AT_LAYER(p_configure,layer) (p_configure->neurons_count[layer] + (p_configure->b_consist_bias != 0 ? 1:0)

double sigmoid(double val)
{
    return (1.0 /(1.0 + exp(-val)));
}

double sigmoid_derivative(double val)
{
    return (val * (1.0 - val));
}
/*
 * TODO: fix bug with mem
 */
NN_BOOL NN_build(NN_configure_t const * const p_configure_params)
{
    int i,j;
    int l_count;

    // allocate mem for neurons
    s_g_NN_neurons = NULL;
    s_g_NN_neurons = malloc (sizeof(double*) * p_configure_params->layer_count);
    for (i = 0; i < p_configure_params->layer_count; ++i)
    {
        s_g_NN_neurons[i] = NULL;
        if (p_configure_params->b_consist_bias && (i < (LAST_LAYER(p_configure_params))))
            s_g_NN_neurons[i] = malloc (sizeof(double) * (p_configure_params->neurons_count[i] + 1));
        else
            s_g_NN_neurons[i] = malloc (sizeof(double) * p_configure_params->neurons_count[i]);
    }

    // allocate mem for weights
    s_g_NN_weights = NULL;
    s_g_NN_weights = malloc(sizeof(double**) * (p_configure_params->layer_count-1));
    for (i = 0; i < LAST_LAYER(p_configure_params); ++i)
    {
        if (p_configure_params->b_consist_bias)
        {
            l_count = (p_configure_params->neurons_count[i] +1);
        }
        else
        {
            l_count = (p_configure_params->neurons_count[i]);
        }
        s_g_NN_weights[i] = malloc(sizeof(double*) * l_count);
        
        for (j = 0; j < l_count; ++j)
        {
            s_g_NN_weights[i][j] = malloc(sizeof(double) * p_configure_params->neurons_count[i+1]);
        }
    }

    // allocate mem for errors    
    s_g_NN_errors = NULL;
    s_g_NN_errors = malloc (sizeof(double*) * (p_configure_params->layer_count-1));
    for (i = 0; i < (p_configure_params->layer_count-1); ++i)
    {
        s_g_NN_errors[i] = malloc (sizeof(double) * p_configure_params->neurons_count[i+1]);
    }

    return NN_TRUE;
}

void NN_initialize_weights_with(NN_configure_t const * const p_configure, double const * const p_weights, const size_t weight_count)
{
    int i,j,j2, w_count = 0;
    printf("0 .. %d\n", (p_configure->layer_count - 1));
    for (i = 0; i < (p_configure->layer_count - 1) && (w_count < weight_count); ++i)
    {
        printf("\t0 .. %d\n", (p_configure->neurons_count[i] + ((p_configure->b_consist_bias != 0) ? 1:0)));
        for (j = 0; j < (p_configure->neurons_count[i] + ((p_configure->b_consist_bias != 0) ? 1:0)) && (w_count < weight_count); ++j)
        {
            printf("\t\t0 .. %d ", p_configure->neurons_count[i+1]);
            for (j2 = 0; j2 < p_configure->neurons_count[i+1] && (w_count < weight_count); ++j2)
            {
                s_g_NN_weights[i][j][j2] = p_weights[w_count];
                ++w_count;
            }
            printf("end\n");
        }
    }
}

void NN_debug_print_weights(NN_configure_t const * const p_configure)
{    
    int i,j,j2, w_count = 0;
    printf("===========\n");
    printf("0 .. %d\n",p_configure->layer_count-1);
    for (i = 0; i < (p_configure->layer_count - 1); ++i)
    {
        printf("\t0 .. %d\n", p_configure->neurons_count[i] + ((p_configure->b_consist_bias != 0) ? 1:0));
        for (j = 0; j < p_configure->neurons_count[i] + ((p_configure->b_consist_bias != 0) ? 1:0); ++j)
        {
            for (j2 = 0; j2 < p_configure->neurons_count[i+1]; ++j2)
            {
                printf("%lf ",s_g_NN_weights[i][j][j2]);
            }
            printf("\n");
        }
    }
    printf("===========\n");
}

void NN_debug_print_errors(NN_configure_t const * const p_configure)
{
    printf("===========\n");
    int layer_i, neuron_i;
    for (layer_i = p_configure->layer_count-2; layer_i >= 0; --layer_i)
    {
        printf("layer %d\n\t",layer_i);
        for (neuron_i = 0; neuron_i < p_configure->neurons_count[layer_i]; ++neuron_i)
        {
            printf("%lf ", layer_i, s_g_NN_errors[layer_i][neuron_i]);
        }
        printf("\n");
    }
    printf("===========\n");
}

void NN_push_values(NN_configure_t const * const p_configure, double const * const p_values, const size_t value_count)
{
    int i;
    for (i = 0; (i < value_count) && (i < p_configure->neurons_count[0]); ++i)
    {
        s_g_NN_neurons[0][i] = p_values[i];
    }
}

int NN_get_result(NN_configure_t const * const p_configure)
{
    int i;
    int max_i = 0;
    double max = s_g_NN_neurons[p_configure->layer_count-1][0];
    for (i = 1; i < p_configure->neurons_count[p_configure->layer_count-1]; ++i)
    {
        if (max < s_g_NN_neurons[p_configure->layer_count-1][i])
        {
            max = s_g_NN_neurons[p_configure->layer_count-1][i];
            max_i = i;
        }
    }
    return max_i;
}

void s_NN_calculate_errors(NN_configure_t const * const p_configure, double const * const p_target)
{
    int layer_i, neuron_i, neuron_j;

    // calculate errors for output layer
    layer_i = LAST_LAYER(p_configure);//p_configure->layer_count - 1; // last layer
    for (neuron_i = 0; neuron_i < p_configure->neurons_count[layer_i]; ++neuron_i)
    {
        s_g_NN_errors[p_configure->layer_count - 2][neuron_i] = (p_target[neuron_i] - s_g_NN_neurons[layer_i][neuron_i]) * sigmoid_derivative(s_g_NN_neurons[layer_i][neuron_i]);
    }
    
    // calculate errors for hidden layer
    for (layer_i = p_configure->layer_count - 3; layer_i >= 0; --layer_i)
    {
        for (neuron_i = 0; neuron_i < p_configure->neurons_count[layer_i]; ++neuron_i)
        {
            s_g_NN_errors[layer_i][neuron_i] = 0.0;
            for (neuron_j = 0; neuron_j < p_configure->neurons_count[layer_i + 1]; ++neuron_j)
            {
                s_g_NN_errors[layer_i][neuron_i] += 
                s_g_NN_errors[layer_i+1][neuron_j] * s_g_NN_weights[layer_i][neuron_i][neuron_j];
            }
            s_g_NN_errors[layer_i][neuron_i] *= sigmoid_derivative( s_g_NN_neurons[layer_i][neuron_i]);
        }
    }
}

void s_NN_update_weights(NN_configure_t const * const p_configure, const double learn_rate)
{
    int layer_i, neuron_i, neuron_j;
    // forwading layers from
    for (layer_i = PRE_LAST_LAYER(p_configure); layer_i >= 0; --layer_i)
    {
        // forwarding 
        for (neuron_i = 0; neuron_i < p_configure->neurons_count[layer_i] + (p_configure->b_consist_bias != 0 ? 1:0); ++neuron_i)
        {
            for (neuron_j = 0; neuron_j < p_configure->neurons_count[layer_i+1]; ++neuron_j)
            {
                s_g_NN_weights[layer_i][neuron_i][neuron_j] += learn_rate * s_g_NN_neurons[layer_i][neuron_i] * s_g_NN_neurons[layer_i][neuron_i];
            }
        }
    }
}

void NN_backpropagate(NN_configure_t const * const p_configure, double const * const p_target, const double learn_rate)
{
    int layer_i, neuron_i, neuron_j;
    layer_i = p_configure->layer_count - 1;
    for (neuron_i = 0; neuron_i < p_configure->neurons_count[layer_i]; ++neuron_i)
            s_g_NN_errors[p_configure->layer_count - 2][neuron_i] = (p_target[neuron_i] - s_g_NN_neurons[layer_i][neuron_i]) * sigmoid_derivative(s_g_NN_neurons[layer_i][neuron_i]);
    
    for (layer_i = PRE_LAST_LAYER(p_configure); layer_i >= 0; --layer_i)
    {
        for (neuron_i = 0; neuron_i < p_configure->neurons_count[layer_i]; ++neuron_i)
        {
            s_g_NN_errors[layer_i][neuron_i] = 0.0;
            for (neuron_j = 0; neuron_j < p_configure->neurons_count[layer_i + 1]; ++neuron_j)
            {
                s_g_NN_neurons[layer_i][neuron_i] = 0;
            }
        }
    }

}

void NN_feed_forward(NN_configure_t const * const p_configure)
{
    int i, j, j2;

    for (i = 1; (i < p_configure->layer_count); ++i)
    {
        for (j = 0; j < p_configure->neurons_count[i]; ++j)
        {
            s_g_NN_neurons[i][j] = 0;

            for (j2 = 0; j2 < p_configure->neurons_count[i-1]; ++j2)
            {
                s_g_NN_neurons[i][j] += s_g_NN_weights[i-1][j2][j] * s_g_NN_neurons[i-1][j2];
            }
            if (p_configure->b_consist_bias)
            {
                s_g_NN_neurons[i][j] += s_g_NN_weights[i-1][p_configure->neurons_count[i-1]][j];
            }
            s_g_NN_neurons[i][j] = sigmoid(s_g_NN_neurons[i][j]);
        }
    }
}