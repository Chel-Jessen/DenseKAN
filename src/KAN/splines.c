#include "splines.h"
#include "math.h"
#include <stdlib.h>
#include <stdio.h>

// Generate a uniform knot vector
void generate_uniform_knot_vector(unsigned int n, unsigned int d, double *knots) {
    for (unsigned int i = 0; i < n + d + 1; i++) {
        knots[i] = (double)i / (n + d);
    }
}

// Basis function
double basis_function(double x) {
    return x / (1 + exp(-x));
}

// Recursive B-spline basis function calculation
double b_spline_basis(unsigned int i, unsigned int k, double x, double *knots) {
    // i == index of the basis function
    // k == order of the basis function
    if (k == 0) {
        return knots[i] <= x && x < knots[i + 1] ? 1.0 : 0.0;
    }
    double coef1 = (x - knots[i]) / (knots[i + k] - knots[i]);
    double coef2 = (knots[i + k + 1] - x) / (knots[i + k + 1] - knots[i + 1]);
    return coef1 * b_spline_basis(i, k - 1, x, knots) + coef2 * b_spline_basis(i + 1, k - 1, x, knots);
}

// Evaluate the B-spline
double calc_spline(double *coeffs, unsigned int grid_size, unsigned int spline_order, double x) {
    unsigned int n = grid_size + spline_order - 1; // == length(coeffs)
    // prevent integer underflow
    if (grid_size == 0 && spline_order == 0) {
        printf("Error: grid_size or spline_order must be greater than 0.\n");
        exit(EXIT_FAILURE);
    }
    double *knots = (double *)calloc(n + spline_order + 1, sizeof(double));
    if (knots == NULL) {
        printf("Memory allocation failed for knots\n");
        exit(EXIT_FAILURE);
    }

    generate_uniform_knot_vector(n, spline_order, knots);

    double result = 0.0;
    for (int i = 0; i < n; i++) {
        result += coeffs[i] * b_spline_basis(i, spline_order, x, knots);
    }

    free(knots);
    return result;
}
