//%%writefile example2p.cu
// 2nd example calculation

#include <math.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include "/include/FBT.cuh"

__host__ __device__ double exact( double qT){ return pow((1.+qT*qT),-1.5);} // exact function (fourier pair)


int main(void)
{
    FBT ogata0_2p = FBT(0, 0, 200, 1); 
    FBT ogata1_2p = FBT(0, 1, 200, 1); 
    FBT ogata2_2p = FBT(0, 2, 200, 1); 
    FBT ogata_def_2p = FBT();

    double qT = 1.;
    double width = 1.;
    double res0 = ogata0_2p.fbt2p(qT , width); 
    double res1 = ogata1_2p.fbt2p(qT , width);
    double res2 = ogata2_2p.fbt2p(qT , width);
    double res_def = ogata_def_2p.fbt2p(qT , width); 

    std::cout << std::setprecision(30) << "Exact = " << exact(qT) << std::endl;
    std::cout << std::setprecision(30) << "Numerical Ogata Opt_t 2p = " << res0 << std::endl;
    std::cout << std::setprecision(30) << "Numerical Ogata UnOpt 2p = " << res1 << std::endl;
    std::cout << std::setprecision(30) << "Numerical Ogata h-fixed(0.05) 2p = " << res2 << std::endl;
    std::cout << std::setprecision(30) << "Numerical Ogata SetDef_ 2p = " << res_def << std::endl;

    return 0;
}
