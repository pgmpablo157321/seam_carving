#include "argparser.h"
#include "display.h"
#include "energy.h"
#include "loader.h"

#include "seam_carving.h"
#include <iostream>
#include <stdio.h>
// Import timing
#include <chrono>

ArgumentParser get_argument_parser(int argc, char *argv[]) {
  ArgumentParser parser;
  parser.add_argument("--n");
  parser.add_argument("--image_path");
  parser.add_argument("--mode");
  parser.add_argument("--direction");
  parser.parse_args(argc, argv);
  return parser;
}

int main(int argc, char *argv[]) {
  // Parse the arguments
  ArgumentParser parser = get_argument_parser(argc, argv);
  std::string image_path = parser.get_argument("--image_path");
  mode color_mode;
  Direction direction;
  int n = stoi(parser.get_argument("--n"));
  if (parser.get_argument("--mode") == "color") {
    color_mode = mode::Color;
  } else {
    color_mode = mode::GrayScale;
  }
  if (parser.get_argument("--direction") == "h") {
    direction = Direction::Horizontal;
  } else {
    direction = Direction::Vertical;
  }
  Loader l(image_path, color_mode);
  int *shape = l.getShape();
  int rows = *(shape + 1), cols = *(shape + 2), channels = *shape;
  float *e = NULL, *img = NULL;

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
  SeamCarving sc(rows, cols, direction, Device::GPU, Device::GPU);
  // Find least energy path, remove it & update dims
  for (int i = 1; i < n + 1; i++) {
    // Find and get least energy path
    sc.findLeastEnergyPath(e);
    // Remove path from energy and image
    sc.removeExternalLeastEnergyPath(img, channels);
    sc.removeExternalLeastEnergyPath(e, 1);

    if (direction == Direction::Horizontal) {
      sc.setRows(rows - i);
      g.setRows(rows - i);
    } else {
      sc.setCols(cols - i);
      g.setCols(cols - i);
    }
  }
  // Display reduced image
  int new_size = 0;
  if (direction == Direction::Horizontal) {
    new_size = (rows - n) * cols * channels;
  } else {
    new_size = rows * (cols - n) * channels;
  }

  float *new_img = (float *)malloc(new_size * sizeof(float));
  cudaMemcpy(new_img, img, new_size * sizeof(float), cudaMemcpyDeviceToHost);
  if (direction == Direction::Horizontal) {
    d.update(new_img, (rows - n), cols, channels, true);
    d.setWindowDims(cols / 2, (rows - n));
  } else {
    d.update(new_img, rows, (cols - n), channels, true);
    d.setWindowDims((cols - n) / 2, rows / 2);
  }

  d.setWindowName("Reduced Image");
  d.displayImage();
  // d.imwrite("assets/sample_noise_reduced_1.jpeg");

  return 0;
}