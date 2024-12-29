#include "seam_carving.h"
#include <chrono>
#include <iostream>
#include <stdio.h>

__global__ void verticalEnergyStep(float *e, float *m, int rows, int cols,
                                   int current_row) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < cols) {
    if (current_row == 0) {
      *(m + current_row * cols + idx) = *(e + current_row * cols + idx);
    } else {
      float aux = *(m + (current_row - 1) * cols + idx);
      if ((idx > 0) && (*(m + (current_row - 1) * cols + (idx - 1)) < aux)) {
        aux = *(m + (current_row - 1) * cols + (idx - 1));
      }
      if ((idx < cols - 1) &&
          (*(m + (current_row - 1) * cols + (idx + 1)) < aux)) {
        aux = *(m + (current_row - 1) * cols + (idx + 1));
      }
      *(m + current_row * cols + idx) = *(e + current_row * cols + idx) + aux;
    }
  }
}

__global__ void horizontalEnergyStep(float *e, float *m, int rows, int cols,
                                     int current_col) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < rows) {
    if (current_col == 0) {
      *(m + idx * cols + current_col) = *(e + idx * cols + current_col);
    } else {
      float aux = *(m + idx * cols + (current_col - 1));
      if ((idx > 0) && (*(m + (idx - 1) * cols + (current_col - 1)) < aux)) {
        aux = *(m + (idx - 1) * cols + (current_col - 1));
      }
      if ((idx < rows - 1) &&
          (*(m + (idx + 1) * cols + (current_col - 1)) < aux)) {
        aux = *(m + (idx + 1) * cols + (current_col - 1));
      }
      *(m + idx * cols + current_col) = *(e + idx * cols + current_col) + aux;
    }
  }
}

__global__ void verticalLeastEnergyPath(float *m, int *path, int rows,
                                        int cols) {
  *(path + rows - 1) = cols - 1;
  for (int j = cols - 1; j >= 0; j--) {
    if (*(m + (rows - 1) * cols + j) <
        *(m + (rows - 1) * cols + *(path + rows - 1))) {
      *(path + rows - 1) = j;
    }
  }
  for (int i = rows - 2; i >= 0; i--) {
    *(path + i) = *(path + (i + 1));
    if (*(path + (i + 1)) > 0 && *(m + i * cols + *(path + (i + 1)) - 1) <
                                     *(m + i * cols + *(path + i))) {
      *(path + i) = *(path + (i + 1)) - 1;
    }
    if (*(path + (i + 1)) < (cols - 1) &&
        (*(m + i * cols + *(path + (i + 1)) + 1) <
         *(m + i * cols + *(path + i)))) {
      *(path + i) = *(path + (i + 1)) + 1;
    }
  }
}

__global__ void horizontalLeastEnergyPath(float *m, int *path, int rows,
                                          int cols) {
  *(path + cols - 1) = rows - 1;
  for (int i = rows - 1; i >= 0; i--) {
    if (*(m + i * cols + (cols - 1)) <
        *(m + *(path + cols - 1) * cols + cols - 1)) {
      *(path + cols - 1) = i;
    }
  }

  for (int j = cols - 2; j >= 0; j--) {
    *(path + j) = *(path + (j + 1));
    if (*(path + (j + 1)) > 0 && *(m + (*(path + (j + 1)) - 1) * cols + j) <
                                     *(m + *(path + j) * cols + j)) {
      *(path + j) = *(path + (j + 1)) - 1;
    }
    if (*(path + (j + 1)) < (rows - 1) &&
        *(m + (*(path + (j + 1)) + 1) * cols + j) <
            *(m + *(path + j) * cols + j)) {
      *(path + j) = *(path + (j + 1)) + 1;
    }
  }
}

__global__ void verticalPathRemove(int *path, int *idx_mat, int *mat,
                                   int *target, int rows, int cols, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row = idx / cols;
  int col = idx % cols;
  if (col < *(path + row)) {
    *(target + idx - row) = *(idx_mat + idx);
  } else if (col > *(path + row)) {
    *(target + idx - (row + 1)) = *(idx_mat + idx);
  } else {
    *(mat + *(idx_mat + idx)) = n;
  }
}

__global__ void horizontalPathRemove(int *path, int *idx_mat, int *mat,
                                     int *target, int rows, int cols, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row = idx / cols;
  int col = idx % cols;
  if (row < *(path + col)) {
    *(target + idx) = *(idx_mat + idx);
  } else if (row > *(path + col)) {
    *(target + idx - cols) = *(idx_mat + idx);
  } else {
    *(mat + *(idx_mat + idx)) = n;
  }
}

__global__ void verticalPathRemoveExternal(int *path, int *src, int *target,
                                           int rows, int cols, int channels) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row = idx / cols;
  int col = idx % cols;
  if (col != *(path + row)) {
    for (int ch = 0; ch < channels; ch++) {
      int offset = ch * rows + row;
      if (col > *(path + row)) {
        offset += 1;
      }
      *(target + ch * rows * cols + idx - offset) =
          *(src + ch * rows * cols + idx);
    }
  }
}

__global__ void verticalPathRemoveExternal(int *path, float *src, float *target,
                                           int rows, int cols, int channels) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row = idx / cols;
  int col = idx % cols;
  if (col != *(path + row)) {
    for (int ch = 0; ch < channels; ch++) {
      int offset = ch * rows + row;
      if (col > *(path + row)) {
        offset += 1;
      }
      *(target + ch * rows * cols + idx - offset) =
          *(src + ch * rows * cols + idx);
    }
  }
}

__global__ void horizontalPathRemoveExternal(int *path, int *src, int *target,
                                             int rows, int cols, int channels) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row = idx / cols;
  int col = idx % cols;
  if (row != *(path + col)) {
    for (int ch = 0; ch < channels; ch++) {
      int offset = ch * cols;
      if (row > *(path + col)) {
        offset += cols;
      }
      *(target + ch * rows * cols + idx - offset) =
          *(src + ch * rows * cols + idx);
    }
  }
}

__global__ void horizontalPathRemoveExternal(int *path, float *src,
                                             float *target, int rows, int cols,
                                             int channels) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row = idx / cols;
  int col = idx % cols;
  if (row != *(path + col)) {
    for (int ch = 0; ch < channels; ch++) {
      int offset = ch * cols;
      if (row > *(path + col)) {
        offset += cols;
      }
      *(target + ch * rows * cols + idx - offset) =
          *(src + ch * rows * cols + idx);
    }
  }
}

SeamCarving::SeamCarving(int rows, int cols, Direction d, Device input_device,
                         Device output_device) {
  this->rows = rows;
  this->cols = cols;
  this->og_rows = rows;
  this->og_cols = cols;
  this->d = d;
  setInputDevice(input_device);
  setOutputDevice(output_device);

  // Allocate space for workflow helper variables
  this->removed_paths = 0;
  int *aux = (int *)malloc(rows * cols * sizeof(int));
  cudaMalloc(&energy_paths, rows * cols * sizeof(float));
  cudaMalloc(&least_energy_path, max(rows, cols) * sizeof(int));
  cudaMalloc(&removal_order_mat, rows * cols * sizeof(int));
  cudaMalloc(&idx_mat, rows * cols * sizeof(int));
  cudaMalloc(&auxiliar_float_buffer, rows * cols * 4 * sizeof(float));
  cudaMalloc(&auxiliar_int_buffer, rows * cols * 4 * sizeof(int));

  // Initial values for idx_max
  // TODO: optimize initialization using cuda
  for (int i = 0; i < rows * cols; i++) {
    *(aux + i) = i;
  }
  cudaMemcpy(idx_mat, aux, rows * cols * sizeof(int), cudaMemcpyHostToDevice);

  // Initial values for
  // TODO: optimize initialization using cuda
  for (int i = 0; i < rows * cols; i++) {
    *(aux + i) = -1;
  }
  cudaMemcpy(removal_order_mat, aux, rows * cols * sizeof(int),
             cudaMemcpyHostToDevice);
};

void SeamCarving::computeEnergyPaths(float *e) {

  // Initialize variable, allocate space and transfer data to GPU
  float *cudaE;
  if (input_device == Device::CPU) {
    cudaMalloc(&cudaE, rows * cols * sizeof(float));
    cudaMemcpy(cudaE, e, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
  } else {
    cudaE = e;
  }

  if (d == Direction::Horizontal) {
    int threads_per_block = 1024;
    int blocks_per_grid = (rows + threads_per_block - 1) / threads_per_block;
    for (int j = 0; j < cols; j++) {
      horizontalEnergyStep<<<blocks_per_grid, threads_per_block>>>(
          cudaE, energy_paths, rows, cols, j);
    }
  } else {
    int threads_per_block = 1024;
    int blocks_per_grid = (cols + threads_per_block - 1) / threads_per_block;
    for (int i = 0; i < rows; i++) {
      verticalEnergyStep<<<blocks_per_grid, threads_per_block>>>(
          cudaE, energy_paths, rows, cols, i);
    }
  }
}

void SeamCarving::findLeastEnergyPath(float *e) {
  computeEnergyPaths(e);
  if (d == Direction::Horizontal) {
    horizontalLeastEnergyPath<<<1, 1>>>(energy_paths, least_energy_path, rows,
                                        cols);
  } else {
    verticalLeastEnergyPath<<<1, 1>>>(energy_paths, least_energy_path, rows,
                                      cols);
  }
}

void SeamCarving::removeLeastEnergyPath() {
  int threads_per_block = 1024;
  int blocks_per_grid =
      (rows * cols + threads_per_block - 1) / threads_per_block;

  if (d == Direction::Horizontal) {
    horizontalPathRemove<<<blocks_per_grid, threads_per_block>>>(
        least_energy_path, idx_mat, removal_order_mat, auxiliar_int_buffer,
        rows, cols, removed_paths);
    cudaMemcpy(idx_mat, auxiliar_int_buffer, (rows - 1) * cols * sizeof(int),
               cudaMemcpyDeviceToDevice);
  } else {
    verticalPathRemove<<<blocks_per_grid, threads_per_block>>>(
        least_energy_path, idx_mat, removal_order_mat, auxiliar_int_buffer,
        rows, cols, removed_paths);
    cudaMemcpy(idx_mat, auxiliar_int_buffer, rows * (cols - 1) * sizeof(int),
               cudaMemcpyDeviceToDevice);
  }
  removed_paths += 1;
}

void SeamCarving::removeExternalLeastEnergyPath(float *target, int channels) {
  int threads_per_block = 1024;
  int blocks_per_grid =
      (rows * cols + threads_per_block - 1) / threads_per_block;
  assert(input_device == Device::GPU);
  assert(channels <= 4);
  if (d == Direction::Horizontal) {
    horizontalPathRemoveExternal<<<blocks_per_grid, threads_per_block>>>(
        least_energy_path, target, auxiliar_float_buffer, rows, cols, channels);
    cudaMemcpy(target, auxiliar_float_buffer,
               (rows - 1) * cols * channels * sizeof(float),
               cudaMemcpyDeviceToDevice);
  } else {
    verticalPathRemoveExternal<<<blocks_per_grid, threads_per_block>>>(
        least_energy_path, target, auxiliar_float_buffer, rows, cols, channels);
    cudaMemcpy(target, auxiliar_float_buffer,
               rows * (cols - 1) * channels * sizeof(float),
               cudaMemcpyDeviceToDevice);
  }
}

void SeamCarving::removeExternalLeastEnergyPath(int *target, int channels) {
  assert(input_device == Device::GPU);
  assert(channels <= 4);
  int threads_per_block = 1024;
  int blocks_per_grid =
      (rows * cols + threads_per_block - 1) / threads_per_block;
  if (d == Direction::Horizontal) {
    horizontalPathRemoveExternal<<<blocks_per_grid, threads_per_block>>>(
        least_energy_path, target, auxiliar_int_buffer, rows, cols, channels);
    cudaMemcpy(target, auxiliar_int_buffer,
               (rows - 1) * cols * channels * sizeof(int),
               cudaMemcpyDeviceToDevice);
  } else {
    verticalPathRemoveExternal<<<blocks_per_grid, threads_per_block>>>(
        least_energy_path, target, auxiliar_int_buffer, rows, cols, channels);
    cudaMemcpy(target, auxiliar_int_buffer,
               rows * (cols - 1) * channels * sizeof(int),
               cudaMemcpyDeviceToDevice);
  }
}

void SeamCarving::setRows(int rows) { this->rows = rows; }

void SeamCarving::setCols(int cols) { this->cols = cols; }

void SeamCarving::getEnergyPaths(float **target) {
  if (getOutputDevice() == Device::CPU) {
    *target = (float *)malloc(rows * cols * sizeof(float));
    cudaMemcpy(*target, energy_paths, rows * cols * sizeof(float),
               cudaMemcpyDeviceToHost);
  } else {
    cudaFree(*target);
    cudaMalloc(target, rows * cols * sizeof(float));
    cudaMemcpy(*target, energy_paths, rows * cols * sizeof(float),
               cudaMemcpyDeviceToDevice);
  }
}

void SeamCarving::getLeastEnergyPath(int **target) {
  int size;
  if (d == Direction::Horizontal) {
    size = cols * sizeof(int);
  } else {
    size = rows * sizeof(int);
  }

  if (getOutputDevice() == Device::CPU) {
    *target = (int *)malloc(size);
    cudaMemcpy(*target, least_energy_path, size, cudaMemcpyDeviceToHost);
  } else {
    cudaMemcpy(*target, least_energy_path, size, cudaMemcpyDeviceToDevice);
  }
}

void SeamCarving::getRemovalOrder(int **target) {
  if (getOutputDevice() == Device::CPU) {
    *target = (int *)malloc(og_rows * og_cols * sizeof(int));
    cudaMemcpy(*target, removal_order_mat, og_rows * og_cols * sizeof(int),
               cudaMemcpyDeviceToHost);
  } else {
    cudaMalloc(target, og_rows * og_cols * sizeof(int));
    cudaMemcpy(*target, removal_order_mat, og_rows * og_cols * sizeof(int),
               cudaMemcpyDeviceToDevice);
  }
}

void SeamCarving::setInputDevice(Device device) { input_device = device; };

void SeamCarving::setOutputDevice(Device device) { output_device = device; };

Device SeamCarving::getInputDevice() { return input_device; };

Device SeamCarving::getOutputDevice() { return output_device; };
