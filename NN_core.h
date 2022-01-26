NN_BOOL NN_build(NN_configure_t const * const p_configure_params);

void NN_initialize_weights_with( NN_configure_t const * const p_configure, double const * const p_weights, const size_t weight_count);

void NN_push_values(NN_configure_t const * const p_configure, double const * const p_values, const size_t value_count);

void NN_feed_forward(NN_configure_t const * const p_configure);

void NN_back_propagate(NN_configure_t const * const p_configure,double const * const target);

int NN_get_result(NN_configure_t const * const p_configure);

void NN_debug_print_weights(NN_configure_t const * const p_configure);


void s_NN_calculate_errors(NN_configure_t const * const p_configure, double const * const p_target);

void s_NN_update_weights(NN_configure_t const * const p_configure, const double learn_rate);

void NN_debug_print_errors(NN_configure_t const * const p_configure);

void NN_debug_print_errors_into_file(NN_configure_t const * const p_configure,char const * const f_file);