# Seam Carving - Cuda Implementation

## Requirements

- opencv (sudo apt-get install -y libopencv-dev)
- cuda and nvcc (NVIDIA Cuda Compiler drive)
- An Nvidia GPU compatible with cuda and Nvidia drivers

## Deployment

Automatic deployment not implemented yet. To build and run the application follow the next steps:

1. Download the target image and place it in the assests folder
2. Put the image filepath in the `main.cu` file. Replace it in the line that contains `<name_of_image_file>`
3. Go into the root folder.
4. Compile the application using nvcc with opencv and x11 (on linux) as dependencies:
```
nvcc -arch=sm_XX src/main.cu src/seam_carving.cu src/display.cpp src/energy.cu src/loader.cpp src/argparser.cpp src/config_parser.cpp -o main $(pkg-config --cflags --libs opencv4) -lX11
```
Note that the sm_XX value is dependent on the GPU you are running

5. Run the application:
```
./main --config src/config.yml
```

## Timing test
1. Follow steps 1, 2 and 3 from the [deployment](#deployment) instructions. Then compile the timing scripts using the following commands:

**GPU Timing:**
```
nvcc -arch=sm_XX test/ref_timing_gpu.cu src/seam_carving.cu src/display.cpp src/energy.cu src/loader.cpp -o timing_gpu $(pkg-config --cflags --libs opencv4) -lX11
```
**CPU Timing:**
```
g++ test/ref_timing_cpu.cpp src/display.cpp src/loader.cpp -o timing_cpu $(pkg-config --cflags --libs opencv4) -lX11
```
2. Run the compiled files:

**GPU Timing:**
```
./timing_gpu
```
**CPU Timing:**
```
./timing_cpu
```
