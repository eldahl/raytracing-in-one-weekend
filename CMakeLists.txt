cmake_minimum_required(VERSION 3.30)

project(rtiow)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS True)

file(GLOB INCLUDES include/*.h include/*.cpp)
file(GLOB SOURCES src/*.h src/*.cpp)

add_executable(${PROJECT_NAME} src/main.cpp ${INCLUDES} ${SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
