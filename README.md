[![GitHub stars](https://img.shields.io/github/stars/Julien-Livet/SymReg.svg)](https://github.com/Julien-Livet/SymReg/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/Julien-Livet/SymReg.svg)](https://github.com/Julien-Livet/SymReg/issues)
[![License](https://img.shields.io/github/license/Julien-Livet/SymReg.svg)](LICENSE)
![CI](https://github.com/Julien-Livet/SymReg/actions/workflows/build.yml/badge.svg)

# SymReg

## General purpose

Let give some symbols like ```x1``` and ```x2```.
The algorithm will consider first the linear combinations ```a*x1+b``` and ```c*x2+d```, then extra start symbolic expressions defined with ```extraExpressions```.
Set ```verbose``` to ```true``` to get some information during regression.
Then the first iteration starts:
- We apply the unary operators defined with ```un_ops```, for example ```e*log(a*x1+b)+f``` and ```g*log(c*x2+d)+h```.
- We apply then the binary operators defined with ```bin_ops```, for example ```i*(a*x1+b+c*x2+d)+j``` same as ```k*x1+l*x2+m```, etc. We can skip some combinations according to symmetric binary operators.

We can define the depth of operators with ```operatorDepth```.
At each combination, we compute an optimal expression. If ```paramValues``` is empty, we search optimal parameters with ```ceres```, else we use a discrete optimizer.
If ```discreteParams``` is ```true```, parameters will be rounded to admissible values after ceres optimization.
We process like that until the computed loss is less than ```epsLoss```.
```eps``` is used to round numeric values and compare with other expressions for example.
It is possible to call user callback during process with ```callback```.

## Installation

```
sudo apt update
sudo apt install -y cmake make libcurl4-openssl-dev libboost-dev libqt5charts5-dev libmlpack-dev libensmallen-dev libarmadillo-dev libstb-dev pybind11-dev graphviz libgraphviz-dev libqt5svg5-dev qtwebengine5-dev build-essential python3-sympy
sudo apt install -y cmake libgoogle-glog-dev libatlas-base-dev libeigen3-dev libsuitesparse-dev
cd ~
mkdir symreg_ws
cd symreg_ws
git clone https://github.com/Julien-Livet/SymReg.git
git clone --recursive https://ceres-solver.googlesource.com/ceres-solver
cd ceres-solver
mkdir build && cd build
cmake .. -DBUILD_TESTING=OFF
make -j$(nproc)
sudo make install
cd ../../SymReg
mkdir build && cd build
cmake ..
make -j$(nproc)
./primes_demo
ctest -V
```
