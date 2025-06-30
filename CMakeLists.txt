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
find_package(OpenMP REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Widgets Charts)

include_directories(include)

set(CMAKE_AUTOMOC ON)

add_executable(primes_demo src/primes_demo.cpp)
target_link_libraries(primes_demo Ceres::ceres OpenMP::OpenMP_CXX Qt5::Widgets Qt5::Charts)

enable_testing()
add_executable(test_symreg tests/test_symreg.cpp)
target_link_libraries(test_symreg Ceres::ceres CURL::libcurl gtest gtest_main OpenMP::OpenMP_CXX)

include(GoogleTest)
gtest_discover_tests(test_symreg)
