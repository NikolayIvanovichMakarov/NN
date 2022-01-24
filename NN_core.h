NN_BOOL NN_build(NN_configure_t const * const p_configure_params);

void NN_initialize_weights_with( NN_configure_t const * const p_configure, double const * const p_weights, const size_t weight_count);

void NN_push_values(NN_configure_t const * const p_configure, double const * const p_values, const size_t value_count);

void feed_forward(NN_configure_t const * const p_configure);

void back_propagate(NN_configure_t const * const p_configure,double const * const target);

int NN_get_result(NN_configure_t const * const p_configure);

void NN_print_weights(NN_configure_t const * const p_configure);