#include "tools.h"
#include <iostream>
#include <stdio.h>

class ImageDisplay {
private:
  // Arrays to store the image
  float *pixel_array;
  uint8_t *img_data;
  cv::Mat mat;
  // Image metadata
  int rows, cols, channels, size;
  // Window options
  std::string window_name;
  int window_height, window_width;
  int screen_height, screen_width;
  bool preserve_ratio;

public:
  ImageDisplay(float *pixels, int rows, int cols, int channels, bool scale);
  void update(float *pixels, int rows, int cols, int channels, bool scale);
  void getScreenResolution();
  void setDefaultDisplayOptions();
  void setWindowName(const std::string &window_name);
  void setWindowDims(int width, int height);
  void preserveWindowRatio(bool ratio);
  void displayImage();
  void scale();
};