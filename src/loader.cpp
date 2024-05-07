#include "loader.h"
#include <iostream>

Loader::Loader(std::string s, mode m) {

  if (m == mode::Color) {
    img = cv::imread(s, cv::IMREAD_COLOR);
  } else if (m == mode::GrayScale) {
    img = cv::imread(s, cv::IMREAD_GRAYSCALE);
  } else {
    return;
  }
  // Load private variables
  rows = img.rows;
  cols = img.cols;
  channels = img.channels();
  (*this).m = m;

  if (m == mode::Color) {
    assert(channels == 3);
  } else if (m == mode::GrayScale) {
    assert(channels == 1);
  }

  // Allocate memory for pixel array
  pixel_array = (float *)malloc(rows * cols * channels * sizeof(float));

  // load image data into pixel array, new shape is (channel, rows, column)
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      int channel_size = rows * cols;
      int idx = i * cols + j;
      int data_idx = i * cols * channels + j * channels;
      for (int k = 0; k < channels; k++) {
        *(pixel_array + channel_size * k + idx) =
            (float)*(img.data + data_idx + k) / 255.;
      }
    }
  }
}

float *Loader::getPixelArray() { return pixel_array; }

uint8_t *Loader::getImgData() { return img.data; }

int *Loader::getShape() {
  int *shape = (int *)malloc(3 * sizeof(int));
  *shape = channels;
  *(shape + 1) = rows;
  *(shape + 2) = cols;
  return shape;
}
