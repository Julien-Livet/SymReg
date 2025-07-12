cmake_minimum_required(VERSION 3.10)
project(SymReg)

#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(
    -march=native
#    -O3 -ffast-math -DNDEBUG
#    -fsanitize=address,undefined -g -O1
  )
endif()

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
find_package(Qt5 REQUIRED COMPONENTS Widgets Charts Svg WebEngineWidgets)
find_package(Armadillo REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GINAC REQUIRED ginac)

include_directories(include)

add_compile_definitions(EIGEN_NO_DEBUG)

set(CMAKE_AUTOMOC ON)

add_executable(primes_demo src/primes_demo.cpp)
target_link_libraries(primes_demo
#  asan ubsan
  pybind11::embed Python3::Python
  Ceres::ceres Qt5::Widgets Qt5::Charts armadillo Qt5::Svg Qt5::WebEngineWidgets gvc cgraph ${GINAC_LIBRARIES})

enable_testing()
add_executable(test_symreg tests/test_symreg.cpp)
target_link_libraries(test_symreg
#  asan ubsan
  pybind11::embed Python3::Python
  Ceres::ceres CURL::libcurl gtest gtest_main armadillo ${GINAC_LIBRARIES})

math(EXPR TIMEOUT "3 * 60 * 60")
add_test(NAME TestSymReg COMMAND test_symreg)
set_tests_properties(TestSymReg PROPERTIES TIMEOUT ${TIMEOUT})

include(GoogleTest)
gtest_discover_tests(test_symreg)
