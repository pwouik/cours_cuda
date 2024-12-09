cmake_minimum_required(VERSION 3.5.1)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(raytracing)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread")
file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cc")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu")

    # Modern CMake
enable_language("CUDA")
add_executable(raytracing
    ${cpu_source_files} ${gpu_source_files})
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )

target_link_libraries(raytracing ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} )
target_link_options(raytracing PRIVATE -lGL -lglut -lGLU -lGLEW)
