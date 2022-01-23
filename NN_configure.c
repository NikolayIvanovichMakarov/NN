#include "NN_configure.h"
#include <stdio.h>

void print_configure_params(NN_configure_t const * const p_configure_params)
{
    int i;
    printf("layer count %d\n", p_configure_params->layer_count);
    if (p_configure_params->b_consist_bias!=0)
    {
        printf("bias consist\n");
    }
    else
    {
        printf("bias not consist");
    }
    for (i = 0; i < p_configure_params->layer_count; ++i)
    {
        printf("neuron count %d on layer %d\n", (i + 1), p_configure_params->neurons_count[i]);
    }
}


void save_configure_params_into_file(char const *const str_filename, NN_configure_t const * const p_configure_params)
{
    int i;
    FILE *f_out = fopen(str_filename,"w");
    
    if (!f_out)

    fprintf("%d\n",p_configure_params->layer_count);
    fprintf("%d\n",p_configure_params->b_consist_bias!=0? 1:0);
    for (i = 0; i < p_configure_params->layer_count; ++i)
    {
        fprintf("%d\n",p_configure_params->neurons_count[i]);
    }
    fclose(f_out);
}