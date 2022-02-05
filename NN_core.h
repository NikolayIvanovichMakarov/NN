#include "NN_types.h"
#include "NN_configure.h"
#include <stdlib.h>


// initialize section
NN_BOOL NN_build(NN_configure_t const * const p_configure_params);
void NN_initialize_weights_with( NN_configure_t const * const p_configure, double const * const p_weights, const size_t weight_count);

// result section
int NN_get_result(NN_configure_t const * const p_configure);
int NN_get_total_weights_count(NN_configure_t const * const p_configure_params);

// function for backpropagation
void NN_push_values(NN_configure_t const * const p_configure, double const * const p_values, const size_t value_count);
void NN_feed_forward(NN_configure_t const * const p_configure);
void s_NN_calculate_errors(NN_configure_t const * const p_configure, double const * const p_target);
void s_NN_update_weights(NN_configure_t const * const p_configure, const double learn_rate);

// debug function
void NN_debug_print_weights(NN_configure_t const * const p_configure);
void NN_debug_print_errors(NN_configure_t const * const p_configure);
void NN_debug_print_neurons_into_file(NN_configure_t const * const p_configure, char const * const f_file);
void NN_debug_print_errors_into_file(NN_configure_t const * const p_configure,char const * const f_file);
void NN_debug_print_weights_into_file(NN_configure_t const * const p_configure,char const * const f_file);