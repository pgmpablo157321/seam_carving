#include "../src/display.h"
#include "../src/energy.h"
#include "../src/loader.h"

#include "../src/seam_carving.h"
#include <iostream>
#include <stdio.h>
// Import timing
#include <chrono>

int main() {
  Loader l("assets/sample_noise_1.jpeg", mode::Color);
  int *shape = l.getShape();
  int rows = *(shape + 1), cols = *(shape + 2), channels = *shape;
  float *e = NULL;
  // Number of removals
  auto global_start = std::chrono::high_resolution_clock::now();
  int n = 900;

  auto start = std::chrono::high_resolution_clock::now();
  GradientEnergy g(rows, cols, channels, l.getPixelArray(), Device::CPU,
                   Device::GPU);
  // Get energy
  g.compute();
  g.getEnergy(&e);
  SeamCarving sc(rows, cols, Direction::Vertical, Device::GPU, Device::GPU);
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "Time to compute energy: " << duration.count() << "\n";
  // Find least energy path, remove it & update dims
  for (int i = 1; i < n + 1; i++) {
    start = std::chrono::high_resolution_clock::now();
    // Find and get least energy path
    sc.findLeastEnergyPath(e);
    // Remove path from energy and image
    sc.removeExternalLeastEnergyPath(e, 1);
    sc.setCols(cols - i);
    stop = std::chrono::high_resolution_clock::now();
    duration =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time to least path: " << duration.count() << "\n";
  }
  g.setInputDevice(Device::GPU);
  g.setCols(cols - n);
  g.setEnergy(e);
  g.setOutputDevice(Device::CPU);
  float *re = NULL;
  g.getEnergy(&re);
  ImageDisplay d(re, rows, cols - n, 1, true);
  auto global_stop = std::chrono::high_resolution_clock::now();
  auto global_duration = std::chrono::duration_cast<std::chrono::microseconds>(
      global_stop - global_start);
  std::cout << "End to end timing: " << global_duration.count() << "\n";
  d.imwrite("assets/reduced_energy_gpu_new.jpeg");
  return 0;
}