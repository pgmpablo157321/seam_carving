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
        (*(energyPaths + i * cols + *(path + (i + 1)) + 1) <
         *(energyPaths + i * cols + *(path + i)))) {
      *(path + i) = *(path + (i + 1)) + 1;
    }
  }
  return path;
}

void removePath(float *energy, int *path, int rows, int cols) {
  int offset = 0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      if (j == *(path + i)) {
        offset += 1;
      } else {
        *(energy + i * cols + j - offset) = *(energy + i * cols + j);
      }
    }
  }
}

int main() {
  Loader l("assets/sample_noise_1.jpeg", mode::Color);
  int *shape = l.getShape();
  int rows = *(shape + 1), cols = *(shape + 2), channels = *shape;
  float *img = l.getPixelArray();
  float eps = 0.0001;
  int n = 80;

  // Compute path CPU
  float *energy = computeEnergy(img, rows, cols, channels);

  // Compute path GPU
  GradientEnergy g(rows, cols, channels, img, Device::CPU, Device::GPU);
  SeamCarving sc(rows, cols, Direction::Vertical, Device::GPU, Device::CPU);
  float *e = NULL;
  int *path_gpu = NULL;
  g.compute();
  g.getEnergy(&e);

  std::cout << "Path removal differences\n";
  for (int i = 0; i < n; i++) {
    // remove path cpu
    int *path_cpu = computeLeastEnergyPath(energy, rows, cols);

    sc.findLeastEnergyPath(e);
    sc.getLeastEnergyPath(&path_gpu);

    for (int l = 0; l < rows; l++) {
      if (*(path_cpu + l) != *(path_gpu + l)) {
        float *energy_gpu = (float *)malloc(rows * cols * sizeof(float));
        cudaMemcpy(energy_gpu, e, rows * cols * sizeof(float),
                   cudaMemcpyDeviceToHost);
        std::cout << "(" << i << "," << l
                  << "): " << " gpu: " << *(path_gpu + l)
                  << " cpu: " << *(path_cpu + l) << "\n";
        std::cout << "(" << "Energy"
                  << "): " << " gpu: "
                  << *(energy_gpu + l * cols + *(path_gpu + l))
                  << " cpu: " << *(energy_gpu + l * cols + *(path_cpu + l))
                  << "\n";
      }
    }
    removePath(energy, path_cpu, rows, cols);
    cols--;
    // remove path gpu
    sc.removeExternalLeastEnergyPath(e, 1);
    sc.setCols(cols);
  }
  float *energy_gpu = (float *)malloc(rows * cols * sizeof(float));
  cudaMemcpy(energy_gpu, e, rows * cols * sizeof(float),
             cudaMemcpyDeviceToHost);
  std::cout << "Resulting differences\n";
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      if (abs(*(energy + i * cols + j) - *(energy_gpu + i * cols + j)) > eps) {
        std::cout << "(" << i << "," << j
                  << "): " << " gpu: " << *(energy_gpu + i * cols + j)
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