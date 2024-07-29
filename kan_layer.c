#include "kan_layer.h"
#include "splines.h"


void call(struct KANDenseLayer *layer, double *x) {
    // the input x has to be of length layer.in
    for (int i = 0; i < layer->out; i++) {
        for (int j = 0; j < layer->in; j++) {
            layer->nodes[i] += (calc_spline(layer->activation_functions[i][j], layer->grid_size, layer->spline_order, x[j]) +
                                basis_function(x[j])) * layer->w;
        }
    }
}

void update_coefficients(struct KANDenseLayer *layer, double ***coeffs) {
    // coeffs need to be the same dimensions as layer.activation_functions
    for (int i = 0; i < layer->out; i++) {
        for (int j = 0; j < layer->in; j++) {
            for (int k = 0; k < layer->grid_size + layer->spline_order - 1; k++) {
                layer->activation_functions[i][j][k] = coeffs[i][j][k];
            }
        }
    }
}

struct KANDenseLayer init_KAN_Layer(unsigned int in, unsigned int out, unsigned int spline_order, unsigned int grid_size, double w) {
    struct KANDenseLayer k = {
            .in = in,
            .out = out,
            .spline_order = spline_order,
            .grid_size = grid_size,
            .w = w,
    };

    // Activation functions are a 3D matrix with dim1=out, dim2=in, dim3=B-Spline coefficients
    k.activation_functions = (double ***) calloc(k.out, sizeof(double **));
    if (k.activation_functions == NULL) {
        printf("Memory allocation failed for activation functions matrix\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < k.out; i++) {
        k.activation_functions[i] = (double **) calloc(k.in,sizeof(double *));
        if (k.activation_functions[i] == NULL) {
            printf("Memory allocation failed for activation functions matrix\n");
            exit(EXIT_FAILURE);
        }

        for (int j = 0; j < k.in; j++) {
            k.activation_functions[i][j] = (double *) calloc((k.grid_size + k.spline_order - 1), sizeof(double));
            if (k.activation_functions[i][j] == NULL) {
                printf("Memory allocation failed for activation functions matrix\n");
                exit(EXIT_FAILURE);
            }
        }
    }

    k.nodes = (double *) calloc(k.out, sizeof(double));

    if (k.nodes == NULL) {
        printf("Memory allocation failed for nodes\n");
        exit(EXIT_FAILURE);
    }

    return k;
}
