
//%%writefile MathFunctions.cuh

#ifndef __MATHFUNCTIONS_CUH__
#define __MATHFUNCTIONS_CUH__


// Function Definition
__host__ __device__ double g_(double x) { return x*exp(-x*x); }


#endif // MATHFUNCTIONS_CUH_
