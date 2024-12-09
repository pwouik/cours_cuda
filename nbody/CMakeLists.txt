cmake_minimum_required(VERSION 3.5.1)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(nbody)
file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.c")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    # Modern CMake
enable_language("CUDA")
add_executable(nbody
    ${cpu_source_files} ${gpu_source_files})
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )

target_link_libraries(nbody ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} )
target_link_options(nbody PRIVATE -lGL -lglut -lGLU -lGLEW)

set_property(TARGET nbody PROPERTY CUDA_ARCHITECTURES native)
