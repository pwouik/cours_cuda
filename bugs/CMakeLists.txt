cmake_minimum_required(VERSION 3.5.1)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(bugs)
file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cc")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    # Modern CMake
enable_language("CUDA")
add_executable(bugs
    ${cpu_source_files} ${gpu_source_files})
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )

target_link_libraries(bugs ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} )
target_link_options(bugs PRIVATE -lGL -lglut -lGLU -lGLEW)

set_property(TARGET bugs PROPERTY CUDA_ARCHITECTURES native)
