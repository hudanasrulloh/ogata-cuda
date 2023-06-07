# Fourier-Bessel Transform with GPU acceleration

This project uses CUDA C++ to accelerate the computation of Fourier-Bessel Transforms. It also includes an implementation of calculating initial first kind of zeros Bessel Transform using Boost library. This code originally comes from ogata quadrature code for calculating FBT in TMD Physics. This versiion improve the code by utilizing GPU for high cost compuatations. For the detail explanation, please visit the original paper : https://inspirehep.net/literature/1739975. Github : https://github.com/UCLA-TMD/Ogata/tree/master. Thank you.

## Prerequisites

You need to have a CUDA-capable GPU, the CUDA toolkit, and Boost library installed on your machine.
This code require CUDA toolkit 8.0 or higher and CUDA driver 10.1 or higher.

## Compilation

Use the provided Makefile to compile the project:

make

## Usage

After compilation, run the Fourier-Bessel Transform program with:

./fourier_bessel_transform

Or run the Boost Math program with:

./boost_math



