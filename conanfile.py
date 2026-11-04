from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout


class SymRegConan(ConanFile):
    name = "symregpy"
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps"
    default_options = {
        "boost/*:header_only": True,
    }

    def layout(self):
        cmake_layout(self)

    def requirements(self):
        self.requires("ceres-solver/2.1.0")
        self.requires("eigen/3.4.0")
        self.requires("pybind11/3.0.1")
        self.requires("boost/1.84.0")

    def generate(self):
        tc = CMakeToolchain(self)
        tc.cache_variables["ONLY_PYBIND"] = True
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build(target="symreg")

    def package(self):
        cmake = CMake(self)
        cmake.install(component="python")
