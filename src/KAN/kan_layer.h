#ifndef KAN_LAYER_H
#define KAN_LAYER_H

#include <stdio.h>
#include <stdlib.h>
#include "splines.h"

struct KANDenseLayer {
    unsigned int in; // size of the input == number of input neurons
    unsigned int out; // number of nodes in the layer

    unsigned int spline_order;
    unsigned int grid_size;
    double w; // Scale factor

    double ***activation_functions; // 3D matrix of dimensions (out) x (in) x (grid_size + spline_order - 1) which holds the coefficients

    double *nodes; // holds the summed post-activation values
};


struct KANDenseLayer init_KAN_Layer(unsigned int in, unsigned int out, unsigned int spline_order, unsigned int grid_size, double w);
void call(struct KANDenseLayer *layer, double *x);
void update_coefficients(struct KANDenseLayer *layer, double ***coeffs);

#endif // KAN_LAYER_H
