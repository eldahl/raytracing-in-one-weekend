mkdir build
cd build
cmake ..
msbuild /P:Configuration=Release rtiow.vcxproj
start /WAIT /B Release/rtiow.exe > ../image.ppm
