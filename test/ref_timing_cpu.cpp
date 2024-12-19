#include "../src/display.h"
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
  Loader l("assets/02_chameleon.jpeg", mode::Color);
  int *shape = l.getShape();
  int rows = *(shape + 1), cols = *(shape + 2), channels = *shape;
  float *img = l.getPixelArray();
  auto global_start = std::chrono::high_resolution_clock::now();
  int n = 900;

  // Compute energy
  auto start = std::chrono::high_resolution_clock::now();
  float *energy = computeEnergy(img, rows, cols, channels);
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "Time to compute energy: " << duration.count() << "\n";

  // Display energy
  // d.setWindowDims(cols / 2, rows);
  // d.setWindowName("Original Energy");
  // d.displayImage();
  for (int i = 1; i < n; i++) {
    start = std::chrono::high_resolution_clock::now();
    // Find and get least energy path
    int *path = computeLeastEnergyPath(energy, rows, cols);
    // Remove path from energy and image
    removePath(energy, path, rows, cols);
    cols--;
    stop = std::chrono::high_resolution_clock::now();
    duration =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time to least path: " << duration.count() << "\n";
  }
  ImageDisplay d(energy, rows, cols, 1, true);
  //  d.setWindowDims(cols / 2, rows);
  //  d.setWindowName("Reduced Energy");
  //  d.displayImage();
  auto global_stop = std::chrono::high_resolution_clock::now();
  auto global_duration = std::chrono::duration_cast<std::chrono::microseconds>(
      global_stop - global_start);
  std::cout << "End to end timing: " << global_duration.count() << "\n";
  d.imwrite("assets/reduced_energy_cpu.jpeg");
}