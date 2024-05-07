#include "tools.h"
#include <iostream>
#include <stdio.h>

enum class mode { Color, GrayScale };

class Loader {
private:
  cv::Mat img;
  float *pixel_array;
  int rows, cols, channels;
  mode m;

public:
  Loader(std::string s, mode m);
  float *getPixelArray();
  uint8_t *getImgData();
  int *getShape();
};