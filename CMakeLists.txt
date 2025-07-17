cmake_minimum_required(VERSION 3.10)
project(SymReg)

set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)

set(NOISE_PERCENTAGE 0.0 CACHE STRING "Noise percentage")

configure_file(cmake/config.h.in config.h @ONLY)

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
find_package(Sym REQUIRED)

include_directories(include)

add_compile_definitions(EIGEN_NO_DEBUG)

set(CMAKE_AUTOMOC ON)

add_executable(primes_demo src/primes_demo.cpp)
target_link_libraries(primes_demo
#  asan ubsan
  pybind11::embed Python3::Python
  Ceres::ceres Qt5::Widgets Qt5::Charts armadillo Qt5::Svg Qt5::WebEngineWidgets gvc cgraph)

enable_testing()
add_executable(test_symreg tests/test_symreg.cpp)
target_link_libraries(test_symreg
#  asan ubsan
  pybind11::embed Python3::Python
  Ceres::ceres CURL::libcurl gtest gtest_main armadillo)
target_include_directories(test_symreg PRIVATE ${CMAKE_BINARY_DIR})

math(EXPR TIMEOUT "3 * 60 * 60")
add_test(NAME TestSymReg COMMAND test_symreg)
set_tests_properties(TestSymReg PROPERTIES TIMEOUT ${TIMEOUT})

include(GoogleTest)
gtest_discover_tests(test_symreg)

install(DIRECTORY include/SymReg DESTINATION include)

add_library(SymReg INTERFACE)
target_include_directories(SymReg INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/SymRegConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/SymRegConfig.cmake
    INSTALL_DESTINATION lib/cmake/SymReg
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/SymRegConfigVersion.cmake
    VERSION 1.0.0
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/SymRegConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/SymRegConfigVersion.cmake
    DESTINATION lib/cmake/SymReg
)

install(TARGETS SymReg EXPORT SymRegTargets)

install(EXPORT SymRegTargets
    NAMESPACE SymReg::
    DESTINATION lib/cmake/SymReg
)
