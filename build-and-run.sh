mkdir -p build
cd build
cmake ..
make
time ./rtiow > ../image.ppm
time ./rtiow_cuda > ../image_cude.ppm
