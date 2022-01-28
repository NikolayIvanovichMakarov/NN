#include <stdlib.h>
#include <stdio.h>
#include "NN_configure.h"
#define MAX_LAYERS_COUNT 100
#define MIN_LAYERS_COUNT 1
#define MAX_NEURONS_COUNT 100
#define MIN_NEURONS_COUNT 1


NN_BOOL NN_parse(char const * const str_filename, NN_configure_t * const p_loading_params);