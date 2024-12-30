#include "../src/display.h"
#include "../src/energy.h"
#include "../src/loader.h"
#include "../src/seam_carving.h"

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

float *computeEnergyPaths(float *energy, int rows, int cols) {
  float *energy_paths = (float *)malloc(rows * cols * sizeof(float));
  float aux = 0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      aux = 0;
      if (i == 0) {
        *(energy_paths + i * cols + j) = *(energy + i * cols + j);
      } else {
        aux = *(energy_paths + (i - 1) * cols + j);
        if ((j > 0) && (*(energy_paths + (i - 1) * cols + j - 1) < aux)) {
          aux = *(energy_paths + (i - 1) * cols + j - 1);
        }
        if ((j < cols - 1) &&
            (*(energy_paths + (i - 1) * cols + j + 1) < aux)) {
          aux = *(energy_paths + (i - 1) * cols + j + 1);
        }
        *(energy_paths + i * cols + j) = aux + *(energy + i * cols + j);
      }
    }
  }
  return energy_paths;
}

int *computeLeastEnergyPath(float *energy, int rows, int cols) {
  float *energyPaths = computeEnergyPaths(energy, rows, cols);
  int *path = (int *)malloc(rows * sizeof(int));

  *(path + rows - 1) = cols - 1;
  for (int j = cols - 1; j >= 0; j--) {
    if (*(energyPaths + (rows - 1) * cols + j) <
        *(energyPaths + (rows - 1) * cols + *(path + rows - 1))) {
      *(path + rows - 1) = j;
    }
  }
  for (int i = rows - 2; i >= 0; i--) {
    *(path + i) = *(path + (i + 1));
    if (*(path + (i + 1)) > 0 &&
        *(energyPaths + i * cols + *(path + (i + 1)) - 1) <
            *(energyPaths + i * cols + *(path + i))) {
      *(path + i) = *(path + (i + 1)) - 1;
    }
    if (*(path + (i + 1)) < (cols - 1) &&
        (*(energyPaths + i * cols + (*(path + (i + 1)) + 1)) <
         *(energyPaths + i * cols + *(path + i)))) {
      *(path + i) = *(path + (i + 1)) + 1;
    }
  }
  return path;
}

int main() {
  Loader l("assets/sample_noise_1.jpeg", mode::Color);
  int *shape = l.getShape();
  int rows = *(shape + 1), cols = *(shape + 2), channels = *shape;
  float *img = l.getPixelArray();
  float eps = 0.0001;

  // Compute path CPU
  float *energy = computeEnergy(img, rows, cols, channels);
  float *paths_cpu = computeEnergyPaths(energy, rows, cols);

  // Compute path GPU
  GradientEnergy g(rows, cols, channels, img, Device::CPU, Device::GPU);
  SeamCarving sc(rows, cols, Direction::Vertical, Device::GPU, Device::GPU);
  float *e = NULL;
  float *paths_gpu = NULL;
  g.compute();
  g.getEnergy(&e);

  sc.computeEnergyPaths(e);
  sc.setOutputDevice(Device::CPU);
  sc.getEnergyPaths(&paths_gpu);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      if (abs(*(paths_gpu + i * cols + j) - *(paths_cpu + i * cols + j)) >
          eps) {
        std::cout << "(" << i << "," << j
                  << "): " << " gpu: " << *(paths_gpu + i * cols + j)
                  << " cpu: " << *(paths_cpu + i * cols + j) << "\n";
      }
    }
  }

  // ImageDisplay d(paths_cpu, rows, cols, 1, true);
  // d.setWindowDims(cols / 2, rows);
  // d.setWindowName("CPU Energy Paths");
  // d.displayImage();

  // ImageDisplay d_gpu(paths_gpu, rows, cols, 1, true);
  // d_gpu.setWindowDims(cols / 2, rows);
  // d_gpu.setWindowName("GPU Energy Paths");
  // d_gpu.displayImage();

  // ImageDisplay d_img(img, rows, cols, channels, true);
  // d_img.setWindowDims(cols / 2, rows);
  // d_img.setWindowName("Image");
  // d_img.displayImage();
}