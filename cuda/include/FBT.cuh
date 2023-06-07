//%%writefile FBT.cuh
//All this code firtly propose compiled on Google Colab 

#define CUB_IGNORE_DEPRECATED_CPP_DIALECT
#ifndef __FBT_CUH__
#define __FBT_CUH__
#include <algorithm>    // std::copy
#include <vector>          // std::vector
#include <functional>      // std::bind
#include <cmath>           // std::abs
#include <iostream>

namespace cmath = std; 

typedef double (*func_t)(double, double);

class FBT {
private:
  double nu; 
  int N; 
  double Q;  
  int option;
  constexpr static double nu_def = 0.0;
  const static int N_def = 300;
  const static int option_def = 0; 
  constexpr static double Q_def = 1.; 
  std::vector<double> jn_zeros0;
  std::vector<double> zeros;
  std::vector<double> xi;
  std::vector<double> w;

  //void acknowledgement();
  //void citation();

  double get_ht(double hu);
  double get_hu(std::function<double (double) > f,  double q);
  double get_hu2p( func_t f , double q, double width);

public:
  FBT(double _nu = nu_def, int option = option_def, int _N = N_def, double _Q = Q_def); // Constructor
  //FBT(double _nu, int _option, int _N, double _Q);
  ~FBT(); // Deconstructor


  double fbt(double q);
  double fbt2p(double q, double width);

};

#endif // __FBT_CUH__

