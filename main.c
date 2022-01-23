#include "NN_configure.h"
#include "NN_types.h"
#include "NN_parsing.h"
#include "NN_core.h"
int main()
{
    NN_configure_t loading_params;
    if (parse("all_ok_file", &loading_params) == NN_NO_PROBLEM)
    {
        print_configure_params(&loading_params);
    }

    return 0;
}