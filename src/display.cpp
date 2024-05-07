
#include "display.h"
#include <iostream>
#include <stdio.h>

#if WIN32
#include <windows.h>
#else
#include <X11/Xlib.h>
#endif

ImageDisplay::ImageDisplay(float *pixels_src, int rows, int cols, int channels,
                           bool scale = true) {
  this->rows = rows;
  this->cols = cols;
  this->channels = channels;
  size = rows * cols * channels;
  // Allocate size
  pixel_array = (float *)malloc(size * sizeof(float));
  // Copy data into pixel array, target shape is (rows, column, channel)
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      for (int k = 0; k < channels; k++) {
        *(pixel_array + i * cols * channels + j * channels + k) =
            *(pixels_src + k * cols * rows + i * cols + j);
      }
    }
  }
  if (scale) {
    this->scale();
  }
  img_data = (uint8_t *)malloc(size * sizeof(uint8_t));
  for (int i = 0; i < size; i++) {
    *(img_data + i) = (uint8_t)round(*(pixel_array + i));
  }
  if (channels == 3) {
    mat = cv::Mat(rows, cols, CV_8UC3, img_data);
  } else if (channels == 1) {
    mat = cv::Mat(rows, cols, CV_8UC1, img_data);
  } else {
    // Handle error
  }
  setDefaultDisplayOptions();
};

void ImageDisplay::update(float *pixels_src, int rows, int cols, int channels,
                          bool scale = true) {
  this->rows = rows;
  this->cols = cols;
  this->channels = channels;
  size = rows * cols * channels;
  // Allocate size
  free(pixel_array);
  pixel_array = (float *)malloc(size * sizeof(float));
  // Copy data into pixel array, new shape is (rows, column, channel)
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      for (int k = 0; k < channels; k++) {
        *(pixel_array + i * cols * channels + j * channels + k) =
            *(pixels_src + k * cols * rows + i * cols + j);
      }
    }
  }
  if (scale) {
    this->scale();
  }
  free(img_data);
  img_data = (uint8_t *)malloc(size * sizeof(uint8_t));
  for (int i = 0; i < size; i++) {
    *(img_data + i) = (uint8_t)round(*(pixel_array + i));
  }
  if (channels == 3) {
    mat = cv::Mat(rows, cols, CV_8UC3, img_data);
  } else if (channels == 1) {
    mat = cv::Mat(rows, cols, CV_8UC1, img_data);
  } else {
    // Handle error
  }
};

void ImageDisplay::getScreenResolution() {
#if WIN32
  width = (int)GetSystemMetrics(SM_CXSCREEN);
  height = (int)GetSystemMetrics(SM_CYSCREEN);
#else
  Display *disp = XOpenDisplay(NULL);
  Screen *scrn = DefaultScreenOfDisplay(disp);
  screen_height = scrn->height;
  screen_width = scrn->width;
#endif
}

void ImageDisplay::setDefaultDisplayOptions() {
  getScreenResolution();
  window_height = 3 * screen_height / 5;
  window_width = 3 * screen_width / 5;
  preserve_ratio = true;
  window_name = "Display window";
};

void ImageDisplay::setWindowName(const std::string &window_name) {
  this->window_name = window_name;
};

void ImageDisplay::setWindowDims(int width, int height) {
  this->window_height = height;
  this->window_width = width;
};

void ImageDisplay::preserveWindowRatio(bool ratio) {
  this->preserve_ratio = ratio;
}

void ImageDisplay::displayImage() {
  if (preserve_ratio) {
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    float ratio = (float)window_width / mat.cols;
    cv::resizeWindow(window_name, window_width, (int)(ratio * mat.rows));
    cv::imshow(window_name, mat);
  } else {
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, window_width, window_height);
    cv::imshow(window_name, mat);
  }

  int k = cv::waitKey(0);
};

void ImageDisplay::scale() {
  float max = 0;
  for (int i = 0; i < size; i++) {
    if (*(pixel_array + i) > max)
      max = *(pixel_array + i);
  }
  if (max > 0) {
    for (int i = 0; i < size; i++) {
      *(pixel_array + i) *= (255. / max);
    }
  }
}
