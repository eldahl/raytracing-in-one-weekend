mkdir -p build
cd build
cmake ..
make
./rtiow > ../image.ppm
