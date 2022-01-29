#include "NN_configure.h"
#include "NN_types.h"
#include "NN_parsing.h"
#include "NN_core.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

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

double benchmark_function (NN_configure_t const * const p_configure, double const * const p_weights, const size_t size)
{
    int test_i, wrong_answer = 0;
    double input[4];
    NN_initialize_weights_with(p_configure, p_weights, size);
    
    for (int test_i  =0; test_i < MAX_SAMPLES; ++test_i)
    {
        input[0] = samples[test_i].health;
        input[1] = samples[test_i].knife;
        input[2] = samples[test_i].gun;
        input[3] = samples[test_i].enemy;

        NN_push_values(p_configure, input, 4);
        NN_feed_forward(p_configure);

        if (NN_get_result(p_configure) != get_max_value(samples[test_i].out,4))
        {
            ++wrong_answer;
        }
    }

    return ((double)wrong_answer)/MAX_SAMPLES; // [0, 18]
}

extern double PI = 3.1415926535897932384626433832795029;

double my_random_func()
{
    double x=-1;
    while ((x<0.0) || (x > 1.0))
    {
        x = ((double)rand())/RAND_MAX;//RAND_MAX+1 // from 0 1
    }
    return x;
}

#define RANDOM() my_random_func()

//!
void generatin_population (double **population, int pop_size, double a, double b, int N)
{
    for (int i=0;i!=pop_size;i++)
    {
        for (int j=0;j!=N;j++)
        {
            population[i][j] = RANDOM()*(b-a)-a;
        }
    }
}

void copy_array(double ** x, double **y, int pop_size, int N)
{
    for (int i=0; i != pop_size; i++)
    {
        memcpy(y[i],x[i], sizeof(x[0]) * N);
    }
}

void indecies_generation(int *r1, int *r2, int *r3, int pop_size)
{
    int a = RANDOM()*(pop_size-1);

    (*r1) = (*r2) = (*r3) = a;

    while ( ((*r1) == (*r2)) || ((*r1) == (*r3)) || ((*r2) == (*r3)))
    {
        (*r1) = RANDOM()*(pop_size-1);
        (*r2) = RANDOM()*(pop_size-1);
        (*r3) = RANDOM()*(pop_size-1);
    }

}

int main(int argc, char **argv)
{
    NN_configure_t loading_params;
    srand(time(0));
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
    int pop_size = 200;              // Размер популяции

    if (argc > 1)
    {
        pop_size = atoi(argv[1]);
    }
    // DE 
    double a = -10.0;               // Границы интервела поиска
    double b = 10.0;                //
    const int N = 31;               // Размерность задачи. Вектор весов
    int FEV = 10000000;             // Кол-во вычислений
    double best_fitness = 50;    // лучшая пригодность
    double **population = malloc(sizeof(double *) * pop_size);
    double **population_new = malloc(sizeof(double *) * pop_size);; // lines

    printf("pop_size = %d\n",pop_size);
    for (int count = 0; count < pop_size; count++)
    {
        population [count] = malloc(sizeof(double) * N);
        population_new [count] = malloc(sizeof(double) * N);
    }

    double *solution = malloc(sizeof(double) * N);
    double *fitness = malloc(sizeof(double) * pop_size);
    double *fitness_new = malloc(sizeof(double) * pop_size);
    double *u = malloc(sizeof(double) * N);
    double test;

    //DE starts
    generatin_population (population, pop_size, a, b, N);
    copy_array(population, population_new, pop_size, N);


    for (int i=0; i!=pop_size; i++)
    {
        for (int j=0; j!=N; j++)
        {
            solution[j]=population[i][j];
        }

        fitness[i] =  benchmark_function(&loading_params, solution, N);
        fitness_new[i] = fitness[i];
        FEV--;
        if (fitness[i]<=best_fitness)
        {
            best_fitness = fitness[i];
        }
    }

    while (FEV>0)
    {

        for (int i=0; i!=pop_size; i++)
        {
            int r1, r2, r3;
            indecies_generation(&r1, &r2, &r3, pop_size);
            double CR = RANDOM()*(0.9-0.1)-0.1;
            double F = RANDOM()*(0.9-0.1)-0.1;
            int jrand = RANDOM()*(N-1);

            for (int j=0; j!=N; j++)
            {
                if (CR<RANDOM() || j == jrand)
                {
                    u[j] = population[i][j]+F*(population[i][j]-population[r1][j])+F*(population[r2][j]-population[r3][j]);
                }
                else
                    u[j] = population[i][j];
            }
            
            for (int j=0; j!=N; j++)
            {
                if (u[j] < a )
                {
                    u[j] = a;
                }
                else if (u[j] > b)
                    u[j] = b;// population[i][j];
            }


            test = benchmark_function(&loading_params,u, N);
            //printf("test = %lf\n",test);
            FEV--;

            if (test<=fitness[i])
            {
                fitness_new[i] = test;
                for (int j=0; j!=N; j++)
                {
                    population_new[i][j]=u[j];
                }
                if (test < best_fitness)
                {
                    best_fitness = test;
                    printf("test = %lf\n",test);
                    //cout<<"FEV: "<<FEV<<" "<<best_fitness<<endl;
                }
            }
        }

        for (int i=0;i!=pop_size; i++)
        {
            for (int j=0;j!=N;j++)
            {
                population[i][j] = population_new[i][j];
            }
            fitness[i] = fitness_new[i];
        }

    }


    return 0;
}