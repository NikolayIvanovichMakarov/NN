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

    if (NN_parse("game.nc", &loading_params) == NN_TRUE)
    {
        printf("parsing ... ok\n");
        print_configure_params(&loading_params);
        if (NN_build(&loading_params) == NN_TRUE)
        {
            printf("building ... ok\n");
            NN_initialize_weights_with(&loading_params, weights, 31);
        }
    }
    printf("weights\n");
    NN_print_weights(&loading_params);
    double input[4];
    for (int i  =0; i < MAX_SAMPLES; ++i)
    {
        input[0] = samples[i].health;
        input[1] = samples[i].knife;
        input[2] = samples[i].gun;
        input[3] = samples[i].enemy;
        NN_push_values(&loading_params, input, 4);

        feed_forward(&loading_params);
        printf("%d) actual = %d\n", i, NN_get_result(&loading_params));
    }


    return 0;
}