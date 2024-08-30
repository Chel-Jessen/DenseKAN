#include <cuda_runtime.h>
#include "../KAN/kan_layer.h"



__device__ double cuda_b_spline_basis(unsigned int i, unsigned int k, double x, const double* knots) {
    if (k == 0) {
        return knots[i] <= x && x < knots[i + 1] ? 1.0 : 0.0;
    }
    double coef1 = (x - knots[i]) / (knots[i + k] - knots[i]);
    double coef2 = (knots[i + k + 1] - x) / (knots[i + k + 1] - knots[i + 1]);
    return coef1 * cuda_b_spline_basis(i, k - 1, x, knots) + coef2 * cuda_b_spline_basis(i + 1, k - 1, x, knots);
}

__global__ void calcSplineValuesKernel(double* x, double* grid, unsigned int grid_size, unsigned int spline_order, double* B, unsigned int batch_size, unsigned int in_size, double* knots) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < in_size && idy < batch_size) {
        for (int k = 0; k < grid_size + spline_order; k++) {
            unsigned int n = grid_size + spline_order - 1;

            double result = 0.0;
            for (int i = 0; i < n; i++) {
                result += grid[idx * (grid_size + 2 * spline_order + 1) + i] * cuda_b_spline_basis(i, spline_order, x[idy * in_size + idx], knots);
            }

            B[(idx * batch_size + idy) * (grid_size + spline_order) + k] = result;
        }
    }
}

__global__ void fitCoefficientsKernel(double* B, double* y, double* coefs, unsigned int batch_size, unsigned int in_size, unsigned int grid_size, unsigned int spline_order, unsigned int out_size, double l2_reg) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < in_size) {
        for (int j = 0; j < out_size; j++) {
            for (int k = 0; k < grid_size + spline_order; k++) {
                double sum = 0.0;
                for (int b = 0; b < batch_size; b++) {
                    sum += B[(idx * batch_size + b) * (grid_size + spline_order) + k] * y[(idx * batch_size + b) * out_size + j];
                }
                coefs[(idx * (grid_size + spline_order) + k) * out_size + j] = sum / (batch_size + l2_reg);
            }
        }
    }
}


extern "C" void fitSplineCoefficients(KANDenseLayer* layer, double* x, double* y, double* grid, unsigned int batch_size, double l2_reg);
void fitSplineCoefficients(KANDenseLayer* layer, double* x, double* y, double* grid, unsigned int batch_size, double l2_reg) {
    unsigned int grid_size = layer->grid_size;
    unsigned int spline_order = layer->spline_order;
    unsigned int in_size = layer->in;
    unsigned int out_size = layer->out;
    unsigned int n = grid_size + spline_order - 1;

    double* h_knots = (double*)calloc(n + spline_order + 1, sizeof(double));
    for (unsigned int i = 0; i < n + spline_order + 1; i++) {
        h_knots[i] = (double)i / (n + spline_order);
    }

    double* d_x;
    double* d_y;
    double* d_grid;
    double* d_B;
    double* d_coefs;
    double* d_knots;

    cudaMalloc(&d_x, batch_size * in_size * sizeof(double));
    cudaMalloc(&d_y, batch_size * in_size * out_size * sizeof(double));
    cudaMalloc(&d_grid, in_size * (grid_size + 2 * spline_order + 1) * sizeof(double));
    cudaMalloc(&d_B, batch_size * in_size * (grid_size + spline_order) * sizeof(double));
    cudaMalloc(&d_coefs, in_size * (grid_size + spline_order) * out_size * sizeof(double));
    cudaMalloc(&d_knots, (n + spline_order + 1) * sizeof(double));

    cudaMemcpy(d_x, x, batch_size * in_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, batch_size * in_size * out_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid, grid, in_size * (grid_size + 2 * spline_order + 1) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_knots, h_knots, (n + spline_order + 1) * sizeof(double), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((in_size + blockDim.x - 1) / blockDim.x, (batch_size + blockDim.y - 1) / blockDim.y);

    calcSplineValuesKernel<<<gridDim, blockDim>>>(d_x, d_grid, grid_size, spline_order, d_B, batch_size, in_size, d_knots);

    dim3 blockDimCoefs(256);
    dim3 gridDimCoefs((in_size + blockDimCoefs.x - 1) / blockDimCoefs.x);

    fitCoefficientsKernel<<<gridDimCoefs, blockDimCoefs>>>(d_B, d_y, d_coefs, batch_size, in_size, grid_size, spline_order, out_size, l2_reg);

    double* h_coefs = (double*)calloc(in_size * (grid_size + spline_order - 1) * out_size, sizeof(double));

    cudaMemcpy(h_coefs, d_coefs, in_size * (grid_size + spline_order - 1) * out_size * sizeof(double), cudaMemcpyDeviceToHost);

    for (unsigned int i = 0; i < out_size; i++) {
        for (unsigned int j = 0; j < in_size; j++) {
            for (unsigned int k = 0; k < grid_size + spline_order - 1; k++) {
                layer->activation_functions[i][j][k] = h_coefs[i * in_size * (grid_size + spline_order - 1) + j * (grid_size + spline_order - 1) + k];
            }
        }
    }

    free(h_coefs);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_grid);
    cudaFree(d_B);
    cudaFree(d_coefs);
    cudaFree(d_knots);

    free(h_knots);
}