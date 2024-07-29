#ifndef DENSEKAN_SPLINES_H
#define DENSEKAN_SPLINES_H

double calc_spline(double *coeffs, unsigned int grid_size, unsigned int spline_order, double x);
void generate_uniform_knot_vector(unsigned int n, unsigned int d, double *knots);
double b_spline_basis(unsigned int i, unsigned int k, double x, double *knots);
double basis_function(double x);

#endif //DENSEKAN_SPLINES_H
