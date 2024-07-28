#!/bin/bash -e

# if it doesn't detect cuda use check that your cuda version supports the HOST compiler (gcc)
# 	https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version/46380601#46380601
# CC=/usr/bin/gcc-10 CXX=/usr/bin/g++-10 cmake -DCMAKE_CUDA_COMPILER:PATH=/usr/local/cuda-11.7/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-10 .. -DCMAKE_BUILD_TYPE=Release


mkdir -p build
cd build

# Pick just 1 of the following -- either Release or Debug
set BUILD_TYPE=Release
#set BUILD_TYPE=Debug

cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} ..
make -j $(nproc)
make package

echo Done!
echo Make sure you install the .deb file:
ls -lh *.deb

cd ..
