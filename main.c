#include "NN_configure.h"
#include "NN_types.h"
#include "NN_parsing.h"
#include "NN_core.h"
#include "NN_learn_dataset.h"

#include <time.h>

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

double benchmark_function (NN_configure_t const * const p_configure, NN_learn_dataset_t const * const p_learndataset, double const * const p_weights, const size_t size)
{
    int test_i, wrong_answer = 0;

    NN_initialize_weights_with(p_configure, p_weights, size);
    for (test_i  =0; test_i < p_learndataset->total_learn_line_count; ++test_i)
    {
        NN_push_values(p_configure, p_learndataset->p_learn_lines[test_i].p_inputs, p_configure->neurons_count[0]);
        NN_feed_forward(p_configure);

        if (NN_get_result(p_configure) != get_max_value(p_learndataset->p_learn_lines[test_i].p_targets, p_configure->neurons_count[p_configure->layer_count-1]))
        {
            ++wrong_answer;
        }
    }

    return ((double)wrong_answer)/p_learndataset->total_learn_line_count; // [0, 18]
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
    for (int i=0;i < pop_size;i++)
    {
        for (int j=0;j < N;j++)
        {
            population[i][j] = RANDOM()*(b-a)-a;
        }
    }
}

void copy_array(double ** x, double **y, int pop_size, int N)
{
    for (int i=0; i < pop_size; i++)
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


typedef enum learn_mode_e
{
    BACK_PROPAGATION,
    DIFFERENTIAL_EVOLUTION = 1
} learn_mode_t;

int main(int argc, char ** argv)
{
    NN_configure_t loading_params;
    NN_learn_dataset_t learn_dataset;
    learn_mode_t learn_mode = DIFFERENTIAL_EVOLUTION;
    NN_BOOL fl_steel_learn = NN_TRUE;

    double input[4];
    int correct_values;
    int j, i;
    char str_file_name_nc[80] = "Data/game.nc";
    char str_file_name_lds[80] = "Data/game.lds";
    char str_file_name_w[80] = "Data/game.w";
    double initial_weights[1024];
    double initial_weights_count = 0;
    double test;

    double a = 0;                   // Границы интервела поиска
    double b = 0;                   //
    int N = 0;                      // Размерность задачи. Вектор весов
    int FEV = 10000000;             // Кол-во вычислений
    double best_fitness = 50;       // лучшая пригодность
    double **population = 0; 
    double **population_new = 0; 
    int pop_size = 50; 
    double *solution ;
    double *u;  
    double *fitness;
    double *fitness_new;
    int r1, r2, r3;
    int count;

    srand(time(0));

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

    printf("choose the learn mode (0 - BP, 1 - DE)\n");
    scanf("%d",&learn_mode);

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
                    printf("correct_values = %lf\n", 1.0-((double)correct_values)/learn_dataset.total_learn_line_count );
                }
                if (correct_values == learn_dataset.total_learn_line_count)
                {
                    printf("correct_values = %lf\n", 1.0-((double)correct_values)/learn_dataset.total_learn_line_count );
                    fl_steel_learn = NN_FALSE;
                }
            }
        break;
        case DIFFERENTIAL_EVOLUTION:
            pop_size = 200;
            a = -10.0;               // Границы интервела поиска
            b = 10.0;                //
            N = NN_get_total_weights_count(&loading_params);               // Размерность задачи. Вектор весов
            FEV = 10000000;             // Кол-во вычислений
            best_fitness = 10000;    // лучшая пригодность
            population = malloc(sizeof(double *) * pop_size);
            population_new = malloc(sizeof(double *) * pop_size);; // lines
            for ( count = 0; count < pop_size; count++)
            {
                population [count] = malloc(sizeof(double) * N);
                population_new [count] = malloc(sizeof(double) * N);
            }

            solution = malloc(sizeof(double) * N);
            fitness = malloc(sizeof(double) * pop_size);
            fitness_new = malloc(sizeof(double) * pop_size);
            u = malloc(sizeof(double) * N);
            
            //DE starts
            printf("generate population ... \n");
            generatin_population (population, pop_size, a, b, N);
            printf("copying array ... \n");
            copy_array(population, population_new, pop_size, N);
            printf("initialize... \n");

            for ( i=0; i < pop_size; i++)
            {
                for ( j=0; j < N; j++)
                {
                    solution[j]=population[i][j];
                }
                fitness[i] = benchmark_function(&loading_params, &learn_dataset, solution, N);
               
                fitness_new[i] = fitness[i];
                FEV--;
                if (fitness[i] < best_fitness)
                {
                    best_fitness = fitness[i];
                    
                }
            }

            while (FEV>0 && best_fitness >0.0)
            {

                for ( i=0; i < pop_size; i++)
                {
                    indecies_generation(&r1,&r2,&r3, pop_size);
                    double CR = RANDOM();//RANDOM()*(0.9-0.1)-0.1;
                    double F = RANDOM();//RANDOM()*(0.9-0.1)-0.1;
                    int jrand = RANDOM()*(N-1);

                    for ( j=0; j < N; j++)
                    {
                        if (CR<RANDOM() || j == jrand)
                        {
                            u[j] = population[i][j]+F*(population[i][j]-population[r1][j])+F*(population[r2][j]-population[r3][j]);
                        }
                        else
                        {
                            u[j] = population[i][j];
                        }
                    }

                    test = benchmark_function(&loading_params, &learn_dataset, u, N);
            
                    FEV--;

                    if (test < fitness[i])
                    {
                        fitness_new[i] = test;
                        for (int j=0; j!=N; j++)
                        {
                            population_new[i][j]=u[j];
                        }
                        if (test < best_fitness)
                        {
                            printf("correct_values = %lf\n",test);
                            best_fitness = test;
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
        break;
    }

    return 0;
}