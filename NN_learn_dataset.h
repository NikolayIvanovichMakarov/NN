#ifndef LEARN_LINE_H
#define LEARN_LINE_H
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "NN_types.h"
#include "NN_configure.h"

typedef struct NN_learn_line_s 
{
    double* p_inputs;
    double* p_targets;
} NN_learn_line_t;

typedef struct NN_learn_dataset_s
{
    int total_learn_line_count;
    NN_learn_line_t *p_learn_lines;
} NN_learn_dataset_t;

NN_learn_line_t* NN_alloc_learn_line(const int input, const int target);

NN_BOOL NN_fill_learn_line(NN_learn_line_t * const p_test_line, double const * const p_input, const size_t input_size, double const * const p_target, const size_t target_size);

void NN_debug_print_learn_dataset(NN_configure_t const * const p_configure, NN_learn_dataset_t const * const p_dataset);

#endif