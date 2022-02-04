#include "NN_configure.h"
#include "NN_types.h"
#include "NN_parsing.h"
#include "NN_core.h"
#include "NN_learn_dataset.h"

typedef struct 
{
    double health;
    double knife;
    double gun;
    double enemy;
    double out[4];
} ELEMENT;

#define MAX_SAMPLES 18

/*H K G E A R W H*/ 
ELEMENT samples[MAX_SAMPLES] = 
{
     { 2.0, 0.0, 0.0, 0.0, {0.0, 0.0, 1.0, 0.0} },
     { 2.0, 0.0, 0.0, 1.0, {0.0, 0.0, 1.0, 0.0} },
     { 2.0, 0.0, 1.0, 1.0, {1.0, 0.0, 0.0, 0.0} },
     { 2.0, 0.0, 1.0, 2.0, {1.0, 0.0, 0.0, 0.0} },
     { 2.0, 1.0, 0.0, 2.0, {0.0, 0.0, 0.0, 1.0} },
     { 2.0, 1.0, 0.0, 1.0, {1.0, 0.0, 0.0, 0.0} },
     { 1.0, 0.0, 0.0, 0.0, {0.0, 0.0, 1.0, 0.0} },
     { 1.0, 0.0, 0.0, 1.0, {0.0, 0.0, 0.0, 1.0} },
     { 1.0, 0.0, 1.0, 1.0, {1.0, 0.0, 0.0, 0.0} },
     { 1.0, 0.0, 1.0, 2.0, {0.0, 0.0, 0.0, 1.0} },
     { 1.0, 1.0, 0.0, 2.0, {0.0, 0.0, 0.0, 1.0} },
     { 1.0, 1.0, 0.0, 1.0, {0.0, 0.0, 0.0, 1.0} },
     { 0.0, 0.0, 0.0, 0.0, {0.0, 0.0, 1.0, 0.0} },
     { 0.0, 0.0, 0.0, 1.0, {0.0, 0.0, 0.0, 1.0} },
     { 0.0, 0.0, 1.0, 1.0, {0.0, 0.0, 0.0, 1.0} },
     { 0.0, 0.0, 1.0, 2.0, {0.0, 1.0, 0.0, 0.0} },
     { 0.0, 1.0, 0.0, 2.0, {0.0, 1.0, 0.0, 0.0} },
     { 0.0, 1.0, 0.0, 1.0, {0.0, 0.0, 0.0, 1.0} }
};

int get_max_value(double const * const p_values, const size_t size)
{
    int max_i = 0;
    int i;
    for (i = 1; i < size; ++i)
    {
        if (p_values[i] > p_values[max_i])
        {
            max_i = i;
        }
    }
    return max_i;
}

typedef enum learn_mode_e
{
    BACK_PROPAGATION,
    DIFFERENTIAL_EVOLUTION = 1
} learn_mode_t;

int main(int argc, char ** argv)
{
    NN_configure_t loading_params;
    NN_learn_dataset_t learn_dataset;
    learn_mode_t learn_mode = BACK_PROPAGATION;
    NN_BOOL fl_steel_learn = NN_TRUE;
    double input[4];
    int correct_values;
    int j, i;
    char str_file_name_nc[80] = "Data/game.nc";
    char str_file_name_lds[80] = "Data/game.lds";
    char str_file_name_w[80] = "Data/game.w";
    double initial_weights[1024];
    double initial_weights_count = 0;


    // analyze arguments
    if (argc > 1)
    {
        strcpy(str_file_name_nc, argv[1]);
    }
    
    if (argc > 2)
    {
        strcpy(str_file_name_lds, argv[2]);
    }
    
    if (argc > 3)
    {
        strcpy(str_file_name_w, argv[3]);
    }
    printf("parsing NN...\n");
    if (NN_parse(str_file_name_nc, &loading_params) != NN_TRUE)
    {
        printf("error on NN file parsing stage\n");
        return 1;
    }

    printf("building NN...\n");
    if (NN_build(&loading_params) != NN_TRUE)
    {
        printf("error on building NN\n");
        return 2;
    }

    printf("parsing learn dataset...\n");
    if (NN_learn_dataset_parse(str_file_name_lds, &loading_params, &learn_dataset ) != NN_TRUE)
    {
        printf("error on learn dataset parsing stage\n");
        return 3;
    }

    printf("parsing weights...\n");
    initial_weights_count = NN_weights_parse(str_file_name_w,initial_weights);
    NN_initialize_weights_with(&loading_params, initial_weights, initial_weights_count);

    printf("\n");
    switch(learn_mode)
    {
        case BACK_PROPAGATION:
            for (j = 0; (j < 1000000) && (fl_steel_learn == NN_TRUE); ++j)
            {
                correct_values = 0;
                for (i = 0; i < learn_dataset.total_learn_line_count && (fl_steel_learn == NN_TRUE); ++i)
                {
                    NN_push_values(&loading_params, learn_dataset.p_learn_lines[i].p_inputs, loading_params.neurons_count[0]);
                    NN_feed_forward(&loading_params);
                    if (NN_get_result(&loading_params) == get_max_value(learn_dataset.p_learn_lines[i].p_targets, loading_params.neurons_count[loading_params.layer_count-1]))
                        ++correct_values;
                    s_NN_calculate_errors(&loading_params, learn_dataset.p_learn_lines[i].p_targets);
                    s_NN_update_weights(&loading_params,0.2);
                }
                if ((j % 1000) == 0)
                {
                    printf("correct_values = %lf\n", ((double)correct_values)/MAX_SAMPLES);
                }
                if (correct_values == learn_dataset.total_learn_line_count)
                {
                    printf("correct_values = %lf\n", ((double)correct_values)/MAX_SAMPLES);
                    fl_steel_learn = NN_FALSE;
                }
            }
        break;
        case DIFFERENTIAL_EVOLUTION:
            for (j = 0; j < 100000; ++j)
            {
                
            }
        break;
    }

    return 0;
}