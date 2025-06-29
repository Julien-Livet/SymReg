cmake_minimum_required(VERSION 3.10)
project(SymReg)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

find_package(Ceres REQUIRED)
find_package(CURL REQUIRED)

include_directories(include)

enable_testing()
add_executable(test_symreg tests/test_symreg.cpp)
target_link_libraries(test_symreg Ceres::ceres CURL::libcurl gtest gtest_main)

include(GoogleTest)
gtest_discover_tests(test_symreg)
