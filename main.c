#include "NN_configure.h"
#include "NN_types.h"
#include "NN_parsing.h"
#include "NN_core.h"

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

int main()
{
    NN_configure_t loading_params;
    double weights[] = 
    {
        -7.681681,  -0.697996,  5.810765, 
        2.092458 ,  4.653695,   -0.260499,
        -6.558963,  4.938535,   -0.005708,
        11.045620,  2.543767,   -5.432555,
        -2.073694,  -3.319861,  8.055421,             //15
        -10.791943, 3.903503,   -7.883254,  9.565421,    //19
        8.816552,   1.536550,   -8.941475,  -1.212408,   //23
        -2.502582,  -8.827701,  4.079397,   7.899403,   //27
        -1.648281,  -1.310184,  0.396202,   -11.899271  //31
    };

    double weights_2[] = 
    {
        -1.1222,  3.697996,  1.810765, 
        2.092458 ,  2.653695,   -2.260499,
        -1.558963,  2.938535,   -0.005708,
        5.145620,  1.543767,   -1.432555,
        -3.073694,  -0.319861,  0.055421,             //15
        -1.791943, 1.903503,   -2.883254,  3.565421,    //19
        1.816552,   0.536550,   -1.941475,  -5.212408,   //23
        -0.502582,  -3.827701,  3.079397,   3.899403,   //27
        -0.648281,  -0.310184,  0.396202,   -0.899271  //31
    };

    if (NN_parse("game.nc", &loading_params) == NN_TRUE)
    {
        printf("parsing ... ok\n");
        print_configure_params(&loading_params);
        if (NN_build(&loading_params) == NN_TRUE)
        {
            printf("building ... ok\n");
            NN_initialize_weights_with(&loading_params, weights_2, 31);
        }
    }
    double input[4];
    int correct_values;
    int j;
    for (j = 0; j < 1; ++j)
    {
        correct_values = 0;
        for (int i  =0; i < 1; ++i)
        {
            input[0] = samples[i].health;
            input[1] = samples[i].knife;
            input[2] = samples[i].gun;
            input[3] = samples[i].enemy;
            NN_push_values(&loading_params, input, 4);
            NN_feed_forward(&loading_params);
            s_NN_calculate_errors(&loading_params,input);
            NN_debug_print_errors_into_file(&loading_params, "errors.txt");
            //NN_debug_print_errors(&loading_params);
            s_NN_update_weights(&loading_params,0.2);
            if (NN_get_result(&loading_params) == get_max_value(samples[i].out,4))
                ++correct_values;
        }
        if ((j % 1000) == 0)
        printf("correct_values = %lf\n", ((double)correct_values)/MAX_SAMPLES);;
    }


    return 0;
}