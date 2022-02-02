#include "NN_learn_dataset.h"
#include "NN_configure.h"

NN_learn_line_t* NN_alloc_learn_line(const int input, const int target)
{
    NN_learn_line_t * p_learn_line = NULL;

    p_learn_line = (NN_learn_line_t*) malloc(sizeof(NN_learn_line_t));
    if (p_learn_line == NULL)
        return NULL;
    
    p_learn_line->p_inputs = NULL;
    p_learn_line->p_targets = NULL;

    p_learn_line->p_inputs = (double*) malloc(sizeof(double) * input);
    if (p_learn_line->p_inputs == NULL)
    {
        free(p_learn_line);
        return NULL;
    }

    p_learn_line->p_targets = (double*) malloc(sizeof(double) * target);
    if (p_learn_line->p_targets)
    {
        free(p_learn_line->p_inputs);
        free(p_learn_line);
        return NULL;
    }
    return p_learn_line;
}

NN_BOOL NN_fill_learn_line(NN_learn_line_t * const p_test_line, double const * const p_input, const size_t input_size, double const * const p_target, const size_t target_size)
{
    if (p_test_line != NULL)
    {
        if ((p_test_line != NULL) && (p_test_line->p_inputs != NULL)  && (p_test_line->p_targets != NULL) && (p_input != NULL) && (p_target != NULL))
        {
            memcpy(p_test_line->p_inputs, p_input, sizeof(double) * input_size);
            memcpy(p_test_line->p_targets, p_target, sizeof(double) * target_size);
            return NN_TRUE;
        }
    }
    return NN_FALSE;
}

void NN_debug_print_learn_dataset(NN_configure_t const * const p_configure, NN_learn_dataset_t const * const p_dataset)
{
    int learn_line_i = 0;
    int lear_line_param_i = 0;
    if (p_dataset == NULL)
        return;
    
    if (p_dataset->total_learn_line_count <= 0)
        return;
    
    for (learn_line_i = 0; learn_line_i < p_dataset->total_learn_line_count; ++learn_line_i)
    {
        if (p_dataset->p_learn_lines[learn_line_i].p_inputs != NULL)
        {
            printf("input { ");
            for (lear_line_param_i = 0; lear_line_param_i < p_configure->neurons_count[0]; ++lear_line_param_i)
            {
                printf("%lf ", p_dataset->p_learn_lines[learn_line_i].p_inputs[lear_line_param_i]);
            }
            printf("} ");
        }

        if (p_dataset->p_learn_lines[learn_line_i].p_targets != NULL)
        {
            printf("target { ");
            for (lear_line_param_i = 0; lear_line_param_i < p_configure->neurons_count[p_configure->layer_count-1]; ++lear_line_param_i)
            {
                printf("%lf ", p_dataset->p_learn_lines[learn_line_i].p_targets[lear_line_param_i]);
            }
            printf("}\n");
        }
    }
}