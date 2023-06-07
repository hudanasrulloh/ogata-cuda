//%%writefile BoostMath.cpp
// This file for producing First kind of zeros Bessel function.
// This because CUDA did not provide BoostMath for directly initialized the first kind .....

#include <boost/math/special_functions/bessel.hpp>
#include <vector>
#include <fstream>
#include <iostream>

void computeBoostMathValues(double nu, int maxN) {
    std::vector<double> jn_zeros0;
    std::vector<double> zeros;
    std::vector<double> xi;
    std::vector<double> Jp1;
    std::vector<double> w;

    try {
        boost::math::cyl_bessel_j_zero(nu, 1, maxN, std::back_inserter(jn_zeros0));
        for (size_t i = 0; i < maxN; i++) {
            if (jn_zeros0[i] == 0) {
                throw std::runtime_error("jn_zeros0[i] is zero.");
            }
            zeros.push_back(jn_zeros0[i]);
            xi.push_back(jn_zeros0[i]/M_PI);
            Jp1.push_back(boost::math::cyl_bessel_j(nu+1.,M_PI*xi[i]));
            if (Jp1[i] != 0) {
              w.push_back(boost::math::cyl_neumann(nu,M_PI*xi[i])/Jp1[i]);
            } else {
              w.push_back(0);
            }  
        }
    }
    catch (std::exception& ex) {
        std::cout << "Thrown exception " << ex.what() << std::endl;
        return; // early return on exception
    }

    // write results to a file
    std::ofstream out_file("precomputed_data.txt");
    for (size_t i = 0; i < maxN; i++) {
        out_file << jn_zeros0[i] << " " << zeros[i] << " " << xi[i] << " " << Jp1[i] << " " << w[i] << "\n";
    }
    out_file.close();
}

int main() {
    // example values for nu and maxN, change these to your needs
    double nu = 0.0;  // default nu = 0, if change we should change the inside jn(n, knots) inside function on FBT code
    const int maxN = 100;  // number of data less than 1000 for better result
    
    computeBoostMathValues(nu, maxN);
    
    return 0;
}

