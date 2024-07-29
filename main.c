#include "stdio.h"
#include "kan_layer.h"
#include "splines.h"


int main() {
    struct KANDenseLayer kan = init_KAN_Layer(2, 5, 3, 5, 1.0);
    double x[2] = {1.3, 1.9};
    call(&kan, x);
    printf("%f\n", basis_function(x[0]));
    printf("%f\n", basis_function(x[1]));
    printf("\n");
    printf("%f\n", basis_function(x[0]) + basis_function(x[1]));
    for (int i = 0; i < kan.out; i++) {
        printf("%f\n", kan.nodes[i]);
    }
    return 0;
}
