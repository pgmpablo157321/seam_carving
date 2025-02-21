#ifndef TOOLS_H
#define TOOLS_H
#if __CUDACC__
#include <cuda_runtime.h>
#endif
#include <opencv2/opencv.hpp>

enum Device { GPU, CPU };

#endif
