
#include "energy.h"
#include <iostream>
#include <stdio.h>

__global__ void gradients(float *img, float *kx, float *ky, float *gx,
                          float *gy, int rows, int cols, int channels) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int kernel_size = 3;
  if (idx < rows * cols * channels) {
    int channel = idx / (rows * cols);
    int row = (idx % (rows * cols)) / cols;
    int col = idx % cols;
    for (int i = 0; i < kernel_size; i++) {
      int offset_rows = i - kernel_size / 2;
      for (int j = 0; j < kernel_size; j++) {
        int offset_cols = i - kernel_size / 2;
        if (((row + offset_rows) >= 0) && ((row + offset_rows) < rows) &&
            ((col + offset_cols) >= 0) && ((col + offset_cols) < cols))
          *(gx + idx) += *(img + channel * rows * cols +
                           (row + offset_rows) * cols + (col + offset_cols)) *
                         (*(kx + kernel_size * i + j));
      }
    }

  } else if (idx < 2 * rows * cols * channels) {
    idx -= rows * cols * channels;
    int channel = idx / (rows * cols);
    int row = (idx % (rows * cols)) / cols;
    int col = idx % cols;
    for (int i = 0; i < kernel_size; i++) {
      int offset_rows = i - kernel_size / 2;
      for (int j = 0; j < kernel_size; j++) {
        int offset_cols = i - kernel_size / 2;
        if (((row + offset_rows) >= 0) && ((row + offset_rows) < rows) &&
            ((col + offset_cols) >= 0) && ((col + offset_cols) < cols))
          *(gy + idx) += *(img + channel * rows * cols +
                           (row + offset_rows) * cols + (col + offset_cols)) *
                         (*(ky + kernel_size * i + j));
      }
    }
  }
}

__global__ void gradientMagnitude(float *gx, float *gy, float *g, int rows,
                                  int cols, int channels) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < rows * cols) {
    float sx = 0, sy = 0;
    for (int i = 0; i < channels; i++) {
      sx += *(gx + i * (rows * cols) + idx);
    }
    for (int i = 0; i < channels; i++) {
      sy += *(gy + i * (rows * cols) + idx);
    }
    *(g + idx) = sqrt(pow(sx, 2) + pow(sy, 2));
  }
}

GradientEnergy::GradientEnergy(int rows, int cols, int channels, float *src,
                               Device input_device, Device output_device) {
  this->rows = rows;
  this->cols = cols;
  this->channels = channels;
  int size = rows * cols * channels * sizeof(float);

  setInputDevice(input_device);
  setOutputDevice(output_device);
  // Copy image to GPU
  if (input_device == Device::CPU) {
    cudaMalloc(&image, size);
    cudaMemcpy(image, src, size, cudaMemcpyHostToDevice);
  } else {
    cudaMalloc(&image, size);
    cudaMemcpy(image, src, size, cudaMemcpyDeviceToDevice);
  }

  // Initialize kernels
  int kernel_size = 3 * 3 * sizeof(float);
  float *aux = (float *)malloc(kernel_size);
  // Init kx
  initKernelX(aux);
  cudaMalloc(&kx, kernel_size);
  cudaMemcpy(kx, aux, kernel_size, cudaMemcpyHostToDevice);
  // Init ky
  initKernelY(aux);
  cudaMalloc(&ky, kernel_size);
  cudaMemcpy(ky, aux, kernel_size, cudaMemcpyHostToDevice);

  // Alloc memory for gx and gy and energy
  cudaMalloc(&gx, size);
  cudaMalloc(&gy, size);
  cudaMalloc(&energy, size / channels);
}

GradientEnergy::~GradientEnergy() {}

void GradientEnergy::compute() {
  int idx_size = rows * cols * channels;
  int threads_per_block = 1024;
  int blocks_per_grid =
      (2 * idx_size + threads_per_block - 1) / threads_per_block;

  gradients<<<blocks_per_grid, threads_per_block>>>(image, kx, ky, gx, gy, rows,
                                                    cols, channels);
  cudaDeviceSynchronize();

  blocks_per_grid =
      ((idx_size / channels) + threads_per_block - 1) / threads_per_block;
  gradientMagnitude<<<blocks_per_grid, threads_per_block>>>(
      gx, gy, energy, rows, cols, channels);
  cudaDeviceSynchronize();
};

void GradientEnergy::getEnergy(float **target) {
  if (getOutputDevice() == Device::CPU) {
    *target = (float *)malloc(rows * cols * sizeof(float));
    cudaMemcpy(*target, energy, rows * cols * sizeof(float),
               cudaMemcpyDeviceToHost);
  } else {
    cudaMalloc(target, rows * cols * sizeof(float));
    cudaMemcpy(*target, energy, rows * cols * sizeof(float),
               cudaMemcpyDeviceToDevice);
  }
};

void GradientEnergy::getSourceImage(float **target) {
  if (getOutputDevice() == Device::CPU) {
    *target = (float *)malloc(rows * cols * channels * sizeof(float));
    cudaMemcpy(*target, image, rows * cols * channels * sizeof(float),
               cudaMemcpyDeviceToHost);
  } else {
    cudaMalloc(target, rows * cols * channels * sizeof(float));
    cudaMemcpy(*target, image, rows * cols * channels * sizeof(float),
               cudaMemcpyDeviceToDevice);
  }
};

void GradientEnergy::getGradientX(float **target) {
  if (getOutputDevice() == Device::CPU) {
    *target = (float *)malloc(rows * cols * channels * sizeof(float));
    cudaMemcpy(*target, gx, rows * cols * channels * sizeof(float),
               cudaMemcpyDeviceToHost);
  } else {
    cudaMalloc(target, rows * cols * channels * sizeof(float));
    cudaMemcpy(*target, gx, rows * cols * channels * sizeof(float),
               cudaMemcpyDeviceToDevice);
  }
};

void GradientEnergy::getGradientY(float **target) {
  if (getOutputDevice() == Device::CPU) {
    *target = (float *)malloc(rows * cols * channels * sizeof(float));
    cudaMemcpy(*target, gy, rows * cols * channels * sizeof(float),
               cudaMemcpyDeviceToHost);
  } else {
    cudaMalloc(target, rows * cols * channels * sizeof(float));
    cudaMemcpy(*target, gx, rows * cols * channels * sizeof(float),
               cudaMemcpyDeviceToDevice);
  }
};

void GradientEnergy::setEnergy(float *e) {
  if (getInputDevice() == Device::CPU) {
    cudaMemcpy(energy, e, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
  } else {
    cudaMemcpy(energy, e, rows * cols * sizeof(float),
               cudaMemcpyDeviceToDevice);
  }
}

void GradientEnergy::setRows(int rows) { this->rows = rows; }
void GradientEnergy::setCols(int cols) { this->cols = cols; }

void GradientEnergy::setInputDevice(Device device) { input_device = device; };

void GradientEnergy::setOutputDevice(Device device) { output_device = device; };

Device GradientEnergy::getInputDevice() { return input_device; };

Device GradientEnergy::getOutputDevice() { return output_device; };
