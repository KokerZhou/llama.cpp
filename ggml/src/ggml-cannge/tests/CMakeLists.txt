cmake_minimum_required(VERSION 3.16)
project(cannge_metadata_tests LANGUAGES CXX)

enable_testing()
add_executable(test-cannge-view test-view.cpp ../view.cpp)
target_compile_features(test-cannge-view PRIVATE cxx_std_17)
target_include_directories(test-cannge-view PRIVATE .. ../../../include)
add_test(NAME cannge-view COMMAND test-cannge-view)
