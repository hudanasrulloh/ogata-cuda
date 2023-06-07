//%%writefile MathFunctions2p.cuh
// I try for 2nd example function for calculation.
// I am following the original Ogata example, for two parameters.
// visit therir documentation: https://ucla-tmd.github.io/Ogata/Usage.html#python-example

#ifndef __MATHFUNCTIONS2PARAM_CUH__
#define __MATHFUNCTIONS2PARAM_CUH__


__host__ __device__ double func_(double x, double width) { return x*exp(-x/width); }




#endif // MATHFUNCTIONS2p_CUH_
