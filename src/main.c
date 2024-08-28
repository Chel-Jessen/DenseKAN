#include "stdlib.h"
#include "KAN/kan_layer.h"


void fitSplineCoefficients(struct KANDenseLayer * kan, double * arr, double * arr1, double * grid, int i, double x);

int main() {
    struct KANDenseLayer kan = init_KAN_Layer(2, 5, 3, 5, 1.0);

    double x[4] = {1.3, 1.9, 2.5, 3.6};

    double y[20] = {
        0.5, 0.7, 0.2, 0.8, 0.3,
        0.6, 0.4, 0.1, 0.2, 0.9,
        0.3, 0.5, 0.7, 0.6, 0.8,
        0.1, 0.3, 0.8, 0.7, 0.4
    };

    double grid[16] = {
        0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0,  3.5, 4.0, 4.5,
        5.0, 5.5, 6.0, 6.5, 7.0,  7.5
    };


    fitSplineCoefficients(&kan, x, y, grid, 2, 0.1);

    call(&kan, x);

    for (int i = 0; i < kan.out; i++) {
        for (int j = 0; j < kan.in; j++) {
            free(kan.activation_functions[i][j]);
        }
        free(kan.activation_functions[i]);
    }
    free(kan.activation_functions);
    free(kan.nodes);

    return 0;
}
