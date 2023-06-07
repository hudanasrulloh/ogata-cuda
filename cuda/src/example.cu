//%%writefile example.cu
// The example calculation I am following the original calculation from Ogata code
// visit their link : https://ucla-tmd.github.io/Ogata/Usage.html#python-example

#include <cmath>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include "/include/FBT.cuh"

__host__ __device__ double exact1( double qT ){ return exp(-qT*qT/4.)/2.;} // test function


int main(void)
{
    FBT ogata0 = FBT(0, 0, 200, 1); 
    FBT ogata1 = FBT(0, 1, 200, 1); 
    FBT ogata2 = FBT(0, 2, 200, 1); 
    FBT ogata_def = FBT();

    double qT = 1.;

    double res0 = ogata0.fbt(qT); 
    double res1 = ogata1.fbt(qT); 
    double res2 = ogata2.fbt(qT); 
    double res_def = ogata_def.fbt(qT); 

    std::cout << std::setprecision(30) << "Exact = " << exact1(qT) << std::endl;
    std::cout << std::setprecision(30) << "Numerical Ogata Opt_t = " << res0 << std::endl;
    std::cout << std::setprecision(30) << "Numerical Ogata UnOpt = " << res1 << std::endl;
    std::cout << std::setprecision(30) << "Numerical Ogata h-fixed (0.05) = " << res2 << std::endl;
    std::cout << std::setprecision(30) << "Numerical Ogata SetDef_ = " << res_def << std::endl;
    //...
    return 0;
}
