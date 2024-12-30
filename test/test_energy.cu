#include "../src/display.h"
#include "../src/energy.h"
#include "../src/loader.h"

#include <iostream>
#include <stdio.h>
// Import timing
#include <chrono>

float *computeEnergy(float *img, int rows, int cols, int channels) {
  float *energy = (float *)malloc(rows * cols * sizeof(float));
  float sx = 0, sy = 0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      sx = 0;
      sy = 0;
      for (int k = 0; k < channels; k++) {
        if (i > 0) {
          sx += 2 * (*(img + k * cols * rows + (i - 1) * cols + j));
          if (j > 0) {
            sx += *(img + k * cols * rows + (i - 1) * cols + (j - 1));
          }
          if (j < cols - 1) {
            sx += *(img + k * cols * rows + (i - 1) * cols + (j + 1));
          }
        }
        if (i < rows - 1) {
          sx -= 2 * (*(img + k * cols * rows + (i + 1) * cols + j));
          if (j > 0) {
            sx -= *(img + k * cols * rows + (i + 1) * cols + (j - 1));
          }
          if (j < cols - 1) {
            sx -= *(img + k * cols * rows + (i + 1) * cols + (j + 1));
          }
        }
        if (j > 0) {
          sy += 2 * (*(img + k * cols * rows + i * cols + (j - 1)));
          if (i > 0) {
            sy += *(img + k * cols * rows + (i - 1) * cols + (j - 1));
          }
          if (i < rows - 1) {
            sy += *(img + k * cols * rows + (i + 1) * cols + (j - 1));
          }
        }
        if (j < cols - 1) {
          sy -= 2 * (*(img + k * cols * rows + i * cols + (j + 1)));
          if (i > 0) {
            sy -= *(img + k * cols * rows + (i - 1) * cols + (j + 1));
          }
          if (i < rows - 1) {
            sy -= *(img + k * cols * rows + (i + 1) * cols + (j + 1));
          }
        }
      }
      *(energy + cols * i + j) = sqrt(pow(sx, 2) + pow(sy, 2));
    }
  }
  return energy;
}

int main() {
  Loader l("assets/sample_noise_1.jpeg", mode::Color);
  int *shape = l.getShape();
  int rows = *(shape + 1), cols = *(shape + 2), channels = *shape;
  float *img = l.getPixelArray();
  float eps = 0.0001;

  // Compute energy CPU
  float *energy = computeEnergy(img, rows, cols, channels);

  // Compute energy GPU
  GradientEnergy g(rows, cols, channels, img, Device::CPU, Device::CPU);
  // Get energy
  float *e = NULL;
  g.compute();
  g.getEnergy(&e);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      if (abs(*(e + i * cols + j) - *(energy + i * cols + j)) > eps) {
        std::cout << "(" << i << "," << j
                  << "): " << " gpu: " << *(e + i * cols + j)
                  << " cpu: " << *(energy + i * cols + j) << "\n";
      }
    }
  }
  // ImageDisplay d(energy, rows, cols, 1, true);
  // d.setWindowDims(cols / 2, rows);
  // d.setWindowName("CPU Energy");
  // d.displayImage();

  // ImageDisplay d_gpu(e, rows, cols, 1, true);
  // d_gpu.setWindowDims(cols / 2, rows);
  // d_gpu.setWindowName("GPU Energy");
  // d_gpu.displayImage();

  // ImageDisplay d_img(img, rows, cols, channels, true);
  // d_img.setWindowDims(cols / 2, rows);
  // d_img.setWindowName("Image");
  // d_img.displayImage();
}