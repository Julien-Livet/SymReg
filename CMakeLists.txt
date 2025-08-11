cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0091 NEW)
project(SymReg)

set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(
#    -O3 -ffast-math -DNDEBUG
#    -fsanitize=address,undefined -g -O1
  )
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
  Sym
  GIT_REPOSITORY https://github.com/Julien-Livet/Sym.git
  GIT_TAG main
)
FetchContent_MakeAvailable(Sym)

find_package(Ceres REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Boost REQUIRED)

include_directories(include)

add_compile_definitions(EIGEN_NO_DEBUG)

add_definitions(-DPYBIND11_DETAILED_ERROR_MESSAGES)
pybind11_add_module(symreg src/python_bindings.cpp)

if (WIN32)
    target_link_libraries(symreg PRIVATE Ceres::ceres Sym Ws2_32 Boost::headers)
else()
    target_link_libraries(symreg PRIVATE Ceres::ceres Sym Boost::headers)
endif()

install(TARGETS symreg LIBRARY DESTINATION .)

if (NOT ONLY_PYBIND)
    find_package(CURL REQUIRED)
    find_package(Qt6 REQUIRED COMPONENTS Widgets Charts Svg)

    set(CMAKE_AUTOMOC ON)

    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.14.0
    )
    FetchContent_MakeAvailable(googletest)

    add_executable(primes_demo src/primes_demo.cpp)

    if (WIN32)
        target_link_libraries(primes_demo
          ${MLPACK_LIBRARIES}
        #  asan ubsan
          pybind11::embed Python3::Python
          Ceres::ceres Qt6::Widgets Qt6::Charts Qt6::Svg gvc cgraph Sym Ws2_32 Boost::headers)
    else()
        target_link_libraries(primes_demo
          ${MLPACK_LIBRARIES}
        #  asan ubsan
          pybind11::embed Python3::Python
          Ceres::ceres Qt6::Widgets Qt6::Charts Qt6::Svg gvc cgraph Sym Boost::headers)
    endif()

    add_executable(benchmark src/benchmark.cpp)

    if (WIN32)
        target_link_libraries(benchmark
        #  asan ubsan
          pybind11::embed Python3::Python
          Ceres::ceres CURL::libcurl Sym Ws2_32 Boost::headers)
    else()
        target_link_libraries(benchmark
        #  asan ubsan
          pybind11::embed Python3::Python
          Ceres::ceres CURL::libcurl Sym Boost::headers)
    endif()

    enable_testing()
    add_executable(test_symreg tests/test_symreg.cpp)

    if (WIN32)
        target_link_libraries(test_symreg
          ${MLPACK_LIBRARIES}
        #  asan ubsan
          pybind11::embed Python3::Python
          Ceres::ceres CURL::libcurl gtest gtest_main Sym Ws2_32 Boost::headers)
    else()
        target_link_libraries(test_symreg
          ${MLPACK_LIBRARIES}
        #  asan ubsan
          pybind11::embed Python3::Python
          Ceres::ceres CURL::libcurl gtest gtest_main Sym Boost::headers)
    endif()

    math(EXPR TIMEOUT "3 * 60 * 60")
    add_test(NAME TestSymReg COMMAND test_symreg)
    set_tests_properties(TestSymReg PROPERTIES TIMEOUT ${TIMEOUT})

    if (WIN32)
    else()
    include(GoogleTest)
    gtest_discover_tests(test_symreg)
    endif()

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
endif()
