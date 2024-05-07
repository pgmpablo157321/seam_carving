# Seam Carving - Cuda Implementation

Currently on development. 


## Requirements

- opencv
- cuda with nvcc
- An Nvidia GPU compatible with cuda and Nvidia drivers

## Deployment

Automatic deployment not implemented yet. To build and run the application follow the next steps:

- Download the target image and place it in the assests folder
- Put the image filepath in the `main.cu` file. Replace it in the line that contains `<name_of_image_file>`
- Go into the src folder:

```
cd src
```
- Compile the application using nvcc with opencv and x11 (on linux) as dependencies:
```
/usr/local/cuda/bin/nvcc -arch=sm_XX -g main seam_carving.cu display.cpp energy.cu loader.cpp -o main $(pkg-config --cflags --libs opencv4) -lX11
```
Note that the sm_XX value is dependent on the GPU you are running
- Run the application:
```
./main
```