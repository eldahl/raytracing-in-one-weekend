cmake_minimum_required(VERSION 3.30)

project(rtiow)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS True)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES native)

file(GLOB INCLUDES include/*.h include/*.hpp include/*.c include/*.cpp)
file(GLOB SOURCES src/*.h src/*.hpp src/*.c src/*.cpp)
list(FILTER SOURCES EXCLUDE REGEX ".*main\\.cpp$")

add_executable(${PROJECT_NAME} src/main.cpp ${INCLUDES} ${SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(rtiow_cuda src/main.cu ${INCLUDES} ${SOURCES})
target_include_directories(rtiow_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
