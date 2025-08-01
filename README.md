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

### Linux

```
sudo apt update
sudo apt install -y cmake make git libcurl4-openssl-dev libboost-dev libqt6charts6-dev libmlpack-dev libensmallen-dev libarmadillo-dev libstb-dev pybind11-dev graphviz libgraphviz-dev libqt6svg6-dev build-essential python3-sympy
sudo apt install -y libgoogle-glog-dev libatlas-base-dev libeigen3-dev libsuitesparse-dev
cd ~
mkdir symreg_ws
cd symreg_ws
git clone https://github.com/Julien-Livet/SymReg.git
git clone --recursive https://ceres-solver.googlesource.com/ceres-solver
cd ceres-solver
mkdir build
cd build
cmake .. -DBUILD_TESTING=OFF
make -j$(nproc)
sudo make install
cd ../..
git clone https://github.com/Julien-Livet/Sym.git
cd Sym
mkdir build
cd build
cmake ..
make -j$(nproc)
sudo make install
cd ../../SymReg
mkdir build
cd build
cmake ..
make -j$(nproc)
sudo make install
./benchmark 0
./primes_demo
ctest -V
time ./test_symreg --gtest_filter=TestSymReg.Line
cd ..
pip install --break-system-packages -e .
```

### Windows with MSYS2 UCRT64

In MSYS2 UCRT64 prompt:
```
pacman -Syu
pacman -Sy git
pacman -Sy mingw-w64-ucrt-x86_64-cmake
pacman -Sy mingw-w64-ucrt-x86_64-ninja
pacman -Sy mingw-w64-ucrt-x86_64-toolchain base-devel
pacman -Sy mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-gdb
pacman -Sy mingw-w64-ucrt-x86_64-boost
pacman -Sy mingw-w64-ucrt-x86_64-qt6
pacman -Sy mingw-w64-ucrt-x86_64-armadillo
pacman -Sy mingw-w64-ucrt-x86_64-ensmallen
pacman -Sy mingw-w64-ucrt-x86_64-python mingw-w64-x86_64-pybind11
pacman -Sy mingw-w64-ucrt-x86_64-graphviz
pacman -Sy mingw-w64-ucrt-x86_64-mlpack
pacman -Sy mingw-w64-ucrt-x86_64-eigen3
pacman -Sy mingw-w64-ucrt-x86_64-ceres-solver
pacman -Sy mingw-w64-ucrt-x86_64-blas
pacman -Sy mingw-w64-ucrt-x86_64-lapack
pacman -Sy mingw-w64-ucrt-x86_64-suitesparse
pacman -Sy mingw-w64-ucrt-x86_64-python-pip
pacman -Sy mingw-w64-ucrt-x86_64-python-pytest
pacman -Sy mingw-w64-ucrt-x86_64-python-numpy
pacman -Sy mingw-w64-ucrt-x86_64-python-sympy
cd ~
mkdir symreg_ws
cd symreg_ws
git clone https://github.com/DrTimothyAldenDavis/SuiteSparse.git
cd SuiteSparse
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/ucrt64 -DBLAS_LIBRARIES=/ucrt64/lib/libblas.dll.a -DLAPACK_LIBRARIES=/ucrt64/lib/liblapack.dll.a ..
ninja -j$(nproc)
ninja install
cd ../..
git clone https://github.com/mlpack/mlpack.git
cd mlpack
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/ucrt64 ..
ninja -j$(nproc)
ninja install
cd ../..
git clone https://github.com/Julien-Livet/SymReg.git
git clone https://github.com/Julien-Livet/Sym.git
cd Sym
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/ucrt64 ..
ninja -j$(nproc)
ninja install
cd ../../SymReg
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/ucrt64 -DBLAS_LIBRARIES=/ucrt64/lib/libblas.dll.a -DLAPACK_LIBRARIES=/ucrt64/lib/liblapack.dll.a -Dpybind11_DIR=/d/Programmes/Python/Python312/Lib/site-packages/pybind11/share/cmake/pybind11 ..
ninja -j$(nproc)
ninja install
cd ..
mkdir symreg
cp build/*pyd symreg
pip install --break-system-packages -e .
cd scripts
pytest -s test_symreg.py
```

Copy following DLL files from MSYS2 UCRT64 bin folder to launch executables from Windows Explorer:
libarmadillo.dll
libopenblas.dll
libblas.dll
libceres-4.dll
libcurl-4.dll
libgcc_s_seh-1.dll
libglog-2.dll
libgvc-6.dll
libstdc++-6.dll
libwinpthread-1.dll
Qt6Charts.dll
Qt6Core.dll
Qt6Gui.dll
Qt6OpenGL.dll
Qt6Svg.dll
Qt6Widgets.dll
zlib1.dll

In CMD prompt:
```
D:
cd D:\Programmes\msys64\home\Julien\symreg_ws\SymReg\build
benchmark.exe 0
primes_demo.exe
ctest -V
test_symreg.exe --gtest_filter=TestSymReg.Line
```

# Benchmark with 0% of noise
|Test name|MSE|Expression time|Test time|Input symbolic expression|Found symbolic expression|
|-|-|-|-|-|-|
|5x1Add7x2Addx3Add8|8.38789e-14|4ms|5ms|`5.2*x1+7.3*x2+x3+8.6`|`5.2*x1 + 7.3*x2 + x3 + 8.6`|
|LinearFit|3.44176e-16|0ms|0ms|`2*x+3`|`2.0*x + 3.0`|
|LogFit|0|1ms|1ms|`2*log(3*x+4)+5`|`2.0*log(3.0*x + 4.0) + 5.0`|
|Test1|4.03717e-15|0ms|0ms|`x**2+x+1`|`x + x**2.0 + 1.0`|
|Test2|4.1359e-24|19ms|19ms|`exp(x)*sin(x)`|`exp(x)*sin(x)`|
|Test3|2.15704e-32|2ms|2ms|`x / (1 + x**2)`|`x/(x**2.0 + 1.0)**1.0`|
|Test4|7.25338e-22|2ms|4ms|`x**2+y**2`|`x**2.0 + y**2.0`|
|Test5|3.35266e-30|21ms|23ms|`log(x)+sin(x)`|`log(x) + sin(x)`|
|Test6|4.84764e-32|991ms|992ms|`exp(-0.5*x**2)/sqrt(2*pi)`|`0.4*exp(-0.5*x**2.0)`|
|PySR|1.21677e-22|1196ms|1438ms|`2.5382*cos(x3)+x0**2-0.5`|`x0**2.0 + 2.5*cos(x3) - 0.5`|
|GPLearn|1.02851e-21|3ms|6ms|`x0**2-x1**2+x1-1`|`x0**2.0 + x1 - 1.0*x1**2.0 - 1.0`|
|x1Mulx2|1.1832e-15|0ms|0ms|`x1*x2`|`x1*x2`|
|Nguyen1|3.28779e-20|1ms|2ms|`x+x**2+x**3`|`1.0*x + 1.0*x**2.0 + 1.0*x**3.0`|
|Nguyen2|4.19683e-20|0ms|1ms|`x+x**2+x**3+x**4`|`1.0*x + 1.0*x**2.0 + 1.0*x**3.0 + 1.0*x**4.0`|
|Nguyen3|2.4968e-20|24ms|50ms|`x+x**2+x**3+x**4+x**5`|`1.0*x + 1.0*x**2.0 + 1.0*x**3.0 + 1.0*x**4.0 + 1.0*x**5.0`|
|Nguyen4|1.24337e-22|92ms|115ms|`x+x**2+x**3+x**4+x**5+x**6`|`1.0*x + 1.0*x**2.0 + 1.0*x**3.0 + 1.0*x**4.0 + 1.0*x**5.0 + 1.0*x**6.0`|
|Nguyen5|1.97215e-31|891ms|898ms|`sin(x**2)*cos(x)-1`|`sin(x**2.0)*cos(x) - 1.0`|
|Nguyen6|1.10934e-31|56ms|60ms|`sin(x)+sin(x+x**2)`|`sin(x) + sin(x + x**2.0)`|
|Nguyen7|1.50377e-30|246ms|250ms|`log(x+1)+log(x**2+1)`|`log(x + 1.0) + log(x**2.0 + 1.0)`|
|Nguyen8|0|0ms|0ms|`sqrt(x)`|`x**0.5`|
|Nguyen9|0|255ms|267ms|`sin(x1)+sin(x2**2)`|`sin(x1) + sin(x2**2.0)`|
|Nguyen10|0|9ms|11ms|`2*sin(x1)*cos(x2)`|`2.0*sin(x1)*cos(x2)`|
|Keijzer10|0|2ms|3ms|`x1**x2`|`x1**x2`|
|d_bacres1|8.92493e-20|0ms|541ms|`20-x-(x*y/(1+0.5*x**2))`|`-1.7*x*y/(0.8*x**2.0 + 1.7)**1.0 - 1.0*x + 20.0`|
|d_bacres2|4.34292e-21|639ms|1059ms|`10-(x*y/(1+0.5*x**2))`|`10.0 - 298.6/(208.0*x**2.0/(x*y)**1.0 + 416.0/(x*y)**1.0)**1.0`|
|d_barmag1|1.42591e-27|4031ms|4409ms|`0.5*sin(x-y)-sin(x)`|`-1.0*sin(x) + 0.5*sin(x - 1.0*y)`|
|d_barmag2|7.61955e-28|4441ms|4805ms|`0.5*sin(y-x)-sin(y)`|`-1.0*sin(y) - 0.5*sin(x - 1.0*y)`|
|d_glider1|1.07305e-25|16170ms|16475ms|`-0.05*x**2-sin(y)`|`-1.0*sin(y)`|
|d_glider2|1.5599e-15|11ms|347ms|`x-cy/x`|`(-1.0*cy + x**2.0)/x**1.0`|
|d_lv1|7.55206e-27|10ms|269ms|`3*x-2*x*y-x**2`|`-2.0*x*y + 3.0*x - 1.0*x**2.0`|
|d_lv2|8.13647e-27|48ms|364ms|`2*y-x*y-y**2`|`-1.0*x*y + 2.0*y - 1.0*y**2.0`|
|d_shearflow1|4.12295e-27|21ms|296ms|`cos(x)*cot(y)`|`cos(x)*cot(y)`|
|d_shearflow2|6.10506e-19|359ms|844ms|`(cy**2+0.1*sy**2)*sx`|`sx*(1.0*cy**2.0 + 0.1*sy**2.0)`|
|d_vdp1|2.31375e-13|58ms|335ms|`10*(y-(1/3*(x**3-x)))`|`3.3*x - 3.3*x**3.0 + 10.0*y`|
|d_vdp2|4.09897e-25|0ms|637ms|`-0.1*x`|`-0.1*x`|

# Benchmark with 0.1% of noise
|Test name|MSE|Expression time|Test time|Input symbolic expression|Found symbolic expression|
|-|-|-|-|-|-|
|5x1Add7x2Addx3Add8|0.00378479|20ms|50ms|`5.2*x1+7.3*x2+x3+8.6`|`5.2*x1 + 7.3*x2 + 1.0*x3 + 8.6`|
|LinearFit|1.14303e-05|0ms|0ms|`2*x+3`|`2.0*x + 3.0`|
|LogFit|0.0022121|17ms|18ms|`2*log(3*x+4)+5`|`2.0*log(3.0*x + 4.0) + 5.0`|
|Test1|0.0644484|14ms|15ms|`x**2+x+1`|`1.0*x + 1.0*x**2.0 + 1.0`|
|Test2|82.7865|196ms|1868ms|`exp(x)*sin(x)`|`exp(x)*sin(x) - 0.1`|
|Test3|1.70665e-06|15ms|2191ms|`x / (1 + x**2)`|`x/(x**2.0 + 1.0)**1.0`|
|Test4|1.25972e-05|265ms|505ms|`x**2+y**2`|`1.0*x**2.0 + 1.0*y**2.0`|
|Test5|0.000162595|712ms|2398ms|`log(x)+sin(x)`|`log(x) + sin(x)`|
|Test6|4.90219e-07|5504ms|13467ms|`exp(-0.5*x**2)/sqrt(2*pi)`|`0.4*exp(-0.5*x**2.0)`|
|PySR|0.000143323|1588ms|6972ms|`2.5382*cos(x3)+x0**2-0.5`|`1.0*x0**2.0 + 2.5*cos(1.0*x3) - 0.4`|
|GPLearn|4.26414e-05|284ms|507ms|`x0**2-x1**2+x1-1`|`1.0*x0**2.0 + 1.0*x1 - 1.0*x1**2.0 - 1.0`|
|x1Mulx2|2.45325e-06|0ms|0ms|`x1*x2`|`1.0*x1*x2`|
|Nguyen1|2.62823e-05|802ms|908ms|`x+x**2+x**3`|`1.0*x + 1.0*x**2.0 + 1.0*x**3.0`|
|Nguyen2|3.18715e-05|652ms|885ms|`x+x**2+x**3+x**4`|`1.0*x + 1.0*x**2.0 + 1.0*x**3.0 + 1.0*x**4.0`|
|Nguyen3|6.71155e-05|935ms|1042ms|`x+x**2+x**3+x**4+x**5`|`1.0*x + 1.0*x**2.0 + 1.0*x**3.0 + 1.0*x**4.0 + 1.0*x**5.0`|
|Nguyen4|4.23057e-05|866ms|977ms|`x+x**2+x**3+x**4+x**5+x**6`|`1.0*x + 1.0*x**2.0 + 1.0*x**3.0 + 1.0*x**4.0 + 1.0*x**5.0 + 1.1*x**6.0 + 0.1*x**7.0 - 0.1*x**9.0`|
|Nguyen5|2.08185e-05|1500ms|1531ms|`sin(x**2)*cos(x)-1`|`sin(x**2.0)*cos(x) - 1.0`|
|Nguyen6|4.39748e-05|219ms|227ms|`sin(x)+sin(x+x**2)`|`sin(x) + sin(x + x**2.0)`|
|Nguyen7|7.69403e-05|401ms|1115ms|`log(x+1)+log(x**2+1)`|`log(x + 1.0) + log(x**2.0 + 1.0)`|
|Nguyen8|3.76766e-05|0ms|1ms|`sqrt(x)`|`x**0.5`|
|Nguyen9|1.51535e-05|339ms|619ms|`sin(x1)+sin(x2**2)`|`sin(x1) + sin(x2**2.0)`|
|Nguyen10|2.14083e-05|36ms|767ms|`2*sin(x1)*cos(x2)`|`2.0*sin(x1)*cos(x2)`|
|Keijzer10|3.67798e-05|6ms|15438ms|`x1**x2`|`x1**x2`|
|d_bacres1|0.0155473|430ms|1293ms|`20-x-(x*y/(1+0.5*x**2))`|`-1.1*x*y/(0.5*x**2.0 - 0.1/(10.0 - 0.7*x)**1.0 + 0.9)**1.0 - 1.02*x + 0.5/(0.5*x**2.0 - 0.1/(10.0 - 0.7*x)**1.0 + 0.9)**1.0 + 20.6`|
|d_bacres2|0.0127714|197ms|541ms|`10-(x*y/(1+0.5*x**2))`|`-0.6*y/(0.3*x + 0.1)**1.0 + 0.1*y/(-3.5*x + 1.6*x**2.0 + 1.9)**1.0 - 0.3/(0.3*x + 0.1)**1.0 + 5.9/(-3.5*x + 1.6*x**2.0 + 1.9)**1.0 + 10.0`|
|d_barmag1|0.00167878|10299ms|40760ms|`0.5*sin(x-y)-sin(x)`|`-1.0*sin(x) + 0.5*sin(x - 1.0*y)`|
|d_barmag2|0.000946167|29806ms|77585ms|`0.5*sin(y-x)-sin(y)`|`-1.0*sin(y) - 0.5*sin(x - 1.0*y)`|
|d_glider1|0.0164986|18617ms|19017ms|`-0.05*x**2-sin(y)`|`-1.0*sin(y)`|
|d_glider2|0.0154252|42ms|336ms|`x-cy/x`|`1.0*(-cy + x**2.0)/x**1.0`|
|d_lv1|0.00831714|7196ms|7750ms|`3*x-2*x*y-x**2`|`-2.0*x*y + 3.0*x - 1.0*x**2.0`|
|d_lv2|0.00166309|4129ms|4496ms|`2*y-x*y-y**2`|`-1.0*x*y + 2.0*y - 1.0*y**2.0`|
|d_shearflow1|0.000936923|118ms|61044ms|`cos(x)*cot(y)`|`cos(x)*cot(y)`|
|d_shearflow2|1.62147|9ms|325ms|`(cy**2+0.1*sy**2)*sx`|`1.1*cy**2.0*sx`|
|d_vdp1|0.0325666|37ms|389ms|`10*(y-(1/3*(x**3-x)))`|`3.3*x - 3.3*x**3.0 + 10.0*y`|
|d_vdp2|2.84792e-06|98ms|414ms|`-0.1*x`|`-0.1*x`|
