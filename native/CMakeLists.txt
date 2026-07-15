cmake_minimum_required(VERSION 3.24)
project(radcounter_embree LANGUAGES CXX)

find_package(embree 4 CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_radcounter_embree src/embree_backend.cpp)
target_compile_features(_radcounter_embree PRIVATE cxx_std_17)
target_link_libraries(_radcounter_embree PRIVATE embree)

if(MSVC)
  target_compile_options(_radcounter_embree PRIVATE /W4 /permissive-)
else()
  target_compile_options(_radcounter_embree PRIVATE -Wall -Wextra -Wpedantic)
endif()

set_target_properties(
  _radcounter_embree
  PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
)
