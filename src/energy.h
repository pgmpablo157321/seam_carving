
#include "tools.h"
#include <iostream>
#include <stdio.h>

class Energy {

public:
  int rows, cols, channels;
  float *image;
  float *energy;
  void compute();
  void getEnergy();
};

class GradientEnergy : public Energy {
private:
  /* data */
  float *gx, *gy;
  float *kx, *ky;
  Device input_device, output_device;

public:
  GradientEnergy(int rows, int cols, int channels, float *src,
                 Device input_device, Device output_device);
  ~GradientEnergy();
  void compute();
  void setEnergy(float *e);
  void setRows(int rows);
  void setCols(int cols);
  void getEnergy(float **target);
  void getSourceImage(float **target);
  void getGradientX(float **target);
  void getGradientY(float **target);
  void setInputDevice(Device device);
  void setOutputDevice(Device device);
  Device getInputDevice();
  Device getOutputDevice();

  static void initKernelX(float *k) {
    *k = 1;
    *(k + 1) = 0;
    *(k + 2) = -1;
    *(k + 3) = 2;
    *(k + 4) = 0;
    *(k + 5) = -2;
    *(k + 6) = 1;
    *(k + 7) = 0;
    *(k + 8) = -1;
  };
  static void initKernelY(float *k) {
    *k = 1;
    *(k + 1) = 2;
    *(k + 2) = 1;
    *(k + 3) = 0;
    *(k + 4) = 0;
    *(k + 5) = 0;
    *(k + 6) = -1;
    *(k + 7) = -2;
    *(k + 8) = -1;
  }
};
