
#include "tools.h"
#include <iostream>
#include <stdio.h>

enum class Direction { Horizontal, Vertical };

class SeamCarving {
private:
  Device input_device, output_device;
  float *energy_paths, *auxiliar_float_buffer;
  int *least_energy_path, *removal_order_mat, *idx_mat, *auxiliar_int_buffer;
  int og_rows, og_cols, rows, cols, removed_paths;
  Direction d;

public:
  SeamCarving(int rows, int cols, Direction d, Device input_device,
              Device output_device);
  void computeEnergyPaths(float *e);
  void findLeastEnergyPath(float *e);
  void removeLeastEnergyPath();
  void removeExternalLeastEnergyPath(float *target, int channels);
  void removeExternalLeastEnergyPath(int *target, int channels);

  // Dimention setters
  void setRows(int rows);
  void setCols(int cols);

  // Device getters and setters
  void setInputDevice(Device device);
  Device getInputDevice();
  void setOutputDevice(Device device);
  Device getOutputDevice();

  // Output of computations
  void getEnergyPaths(float **target);
  void getLeastEnergyPath(int **target);
  void getRemovalOrder(int **target);
};