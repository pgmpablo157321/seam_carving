#include "display.h"
#include "energy.h"
#include "loader.h"

#include "seam_carving.h"
#include <iostream>
#include <stdio.h>
// Import timing
#include <chrono>

int main() {
  Loader l("assets/02_chameleon.jpeg", mode::Color);
  int *shape = l.getShape();
  int rows = *(shape + 1), cols = *(shape + 2), channels = *shape;
  float *e = NULL, *img = NULL;
  // Number of removals
  int n = 200;

  GradientEnergy g(rows, cols, channels, l.getPixelArray(), Device::CPU,
                   Device::CPU);
  g.getSourceImage(&img);

  // Display original image
  ImageDisplay d(img, rows, cols, channels, true);
  d.setWindowDims(cols / 2, rows);
  d.setWindowName("Original Image");
  d.displayImage();
  // Get energy
  g.setOutputDevice(Device::GPU);
  g.getSourceImage(&img);
  g.compute();
  g.getEnergy(&e);
  SeamCarving sc(rows, cols, Direction::Horizontal, Device::GPU, Device::GPU);
  // Find least energy path, remove it & update dims
  for (int i = 1; i < n + 1; i++) {
    // Find and get least energy path
    sc.findLeastEnergyPath(e);
    // Remove path from energy and image
    sc.removeExternalLeastEnergyPath(img, channels);
    sc.removeExternalLeastEnergyPath(e, 1);

    sc.setRows(rows - i);
    g.setRows(rows - i);
  }
  // Display reduced image
  float *new_img =
      (float *)malloc((rows - n) * cols * channels * sizeof(float));
  cudaMemcpy(new_img, img, (rows - n) * cols * channels * sizeof(float),
             cudaMemcpyDeviceToHost);
  d.update(new_img, (rows - n), cols, channels, true);
  d.setWindowDims(cols / 2, (rows - n));
  d.setWindowName("Reduced Image");
  d.displayImage();

  return 0;
}