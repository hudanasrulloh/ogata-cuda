//%%writefile FBT-v2-2p.cu

// I separated for calculating the two parameters input. 
// Please give suggestion if any more effective way than separate the file

#define _USE_MATH_DEFINES // using M_PI for pi

#include <cmath> // abs
#include <mwaitxintrin.h>
#include <iostream>
#include <fstream>
#include <functional> 
#include <cuda_runtime.h>
#include "/include/FBT.cuh"
#include "/include/MathFunctions2Param.cuh" // include MathFunctions for 2 Parameters (2nd example)
#include <cub/cub.cuh>
#include <cub/block/block_reduce.cuh>
#include <numeric>

#include <boost/math/tools/minima.hpp>
/*
//acknowledgement
void FBT::acknowledgement(){
    std::cout << "###############################################################################" << std::endl;
    std::cout << "#                                                                             #" << std::endl;
    std::cout << "#                Fast Bessel Transform (FBT) for TMDs                         #" << std::endl;
    std::cout << "#     Zhongbo Kang, Alexei Prokudin, Nobuo Sato, John Terry                   #" << std::endl;
    std::cout << "#                   Please cite Kang:2019ctl                                  #" << std::endl;
    std::cout << "#                  nu is Bessel function order                                #" << std::endl;
    std::cout << "#                  option = 0,1,2  (modified, unmodified, fixed h) Ogata      #" << std::endl;
    std::cout << "#                  N is number of function calls                              #" << std::endl;
    std::cout << "#                  Q initial guess where function has maximum                 #" << std::endl;
    std::cout << "#                  ::::::::::::::::::::::::::::::                             #" << std::endl;
    std::cout << "#                  :::::::::::::::::::::::::::::                              #" << std::endl;
    std::cout << "#                  Inspired by all those above guys                           #" << std::endl;
    std::cout << "#                  MODIFIED by Kasava (huda nasrulloh)                        #" << std::endl;
    std::cout << "#                  USTC, Modern Physics, 2023                                 #" << std::endl;
    std::cout << "#                  Maximum N for stability less than 500                      #" << std::endl;
    std::cout << "#                  CUDA Toolkit 8.0 or higher                                 #" << std::endl;
    std::cout << "#                  CUDA version 10.1 or higher                                #" << std::endl;
    std::cout << "#                  --- Thank you using this platform ----                     #" << std::endl;
    std::cout << "##--------- Optimized with GPU ---- By HN ----- 2023  ------------------------#" << std::endl;
    std::cout << "###############################################################################" << std::endl;
};

//citation
void FBT::citation(){
    std::cout << "###############################################################################" << std::endl;
    std::cout << "#                     Thank you for using FBT!                                #" << std::endl;
    std::cout << "# Please, cite Kang:2019ctl if used for a publication                         #" << std::endl;
    std::cout << "# --------------  on process for me Kasava (huda nasrulloh)  -------          #" << std::endl;
    std::cout << "###############################################################################" << std::endl;
};
*/
// Deconstructor
FBT::~FBT(){
  //jn_zeros0.~vector<double>();
  //citation();
};


__device__ double get_psi_device(double t){
  return ( t )*tanh( M_PI/2.* sinh( t ) );
};

__device__ double get_psip_device(double t){
  return M_PI*t*( -pow(tanh( M_PI*sinh(t)/2.),2) + 1.)*cosh(t)/2. + tanh(M_PI*sinh(t)/2.);
};


__device__ double f_for_ogata_device2p(double x, double width, double q){
  return func_(x/q, width) / q;
}


// Constructor
FBT::FBT(double _nu, int _option, int _N, double _Q){
    if( _nu >= 0.){
    this->nu     = _nu;
    } else {
        std::cerr << " The value of nu = " << _nu << " is not supported." << std::endl;
        std::cerr << " Falling back to default  nu = " << FBT::nu_def << std::endl;
        this->nu     = FBT::nu_def;
    }

    if( _N >= 1){
        this->N     = _N;
    } else {
        std::cerr << " The value of N = " << _N << " is not supported." << std::endl;
        std::cerr << " Falling back to default  N = "  << FBT::N_def <<std::endl;
        this->N     = FBT::N_def;
    }

    if( _Q > 0){
        this->Q     = _Q;
    } else {
        std::cerr << " The value of Q = " << _Q << " is not supported." << std::endl;
        std::cerr << " Falling back to default  Q = "  << FBT::Q_def <<std::endl;
        this->Q     = FBT::Q_def;
    }

    if( _option <= 2 && _option >= 0){
        this->option     = _option;
    } else {
        std::cerr << " The value of option = " << _option << " is not supported." << std::endl;
        std::cerr << " Falling back to default  option = "  << FBT::option_def <<std::endl;
        this->option     = FBT::option_def;
    }

    // read precomputed data from a file
    std::string filename = "precomputed_data.txt";
    std::ifstream in_file(filename);
    std::vector<double> jn_zeros0_host, zeros_host, xi_host, Jp1_host, w_host;
    if (in_file.is_open()) {
        double a, b, c, d, e;
        while (in_file >> a >> b >> c >> d >> e) {
            jn_zeros0_host.push_back(a);
            zeros_host.push_back(b);
            xi_host.push_back(c);
            Jp1_host.push_back(d);
            w_host.push_back(e);
        }
    in_file.close();
    } else {
        std::cerr << "Unable to open file " << filename << std::endl;
    // handle error
    }
    this->jn_zeros0 = jn_zeros0_host;
    this->zeros = zeros_host;
    this-> xi = xi_host ;
    this-> w = w_host;

    //acknowledgement();
};


__global__ void ogatau_kernel2p(int N, double q, double width, double* h, double* xi, double* w, double* d_block_results) {
    typedef cub::BlockReduce<double, 128> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    double thread_data = 0;
    if (i < N) {
        int nu = 0;
        double knots = xi[i]*(*h);
        double F = f_for_ogata_device2p(knots, width, q)*jn(nu, knots);
        thread_data = (*h)*w[i]*F;  // Calculate the result in each thread
    }

    double block_sum = BlockReduce(temp_storage).Sum(thread_data);

    // Only the first thread in the block writes to global memory
    if (threadIdx.x == 0) {
        d_block_results[blockIdx.x] = block_sum;
    }
}

__global__ void ogatat_kernel2p(int N, double q, double width, double* h, double* xi, double* w, double* d_block_results) {
    typedef cub::BlockReduce<double, 128> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    double thread_data = 0;
    if (i < N) {
        int nu = 0;
        double knots = M_PI/(*h)*get_psi_device((*h)*xi[i]);
        double Jnu = jn(nu, knots); // Using j0() here, assuming nu = 0
        double F = f_for_ogata_device2p(knots, width, q);
        double psip = std::isnan(get_psip_device((*h)*xi[i])) ? 1. : get_psip_device((*h)*xi[i]);
        thread_data = M_PI * F * Jnu * psip * w[i];  // Calculate the result in each thread
    
    }

    double block_sum = BlockReduce(temp_storage).Sum(thread_data);

    // Only the first thread in the block writes to global memory
    if (threadIdx.x == 0) {
        d_block_results[blockIdx.x] = block_sum;
        
    }
}


// now use g as a function pointer
double f_for_get_hu2p(double x, double width, func_t g, double q) {
   return -abs(x*g(x/q, width)); 
}


//"""Determines the untransformed hu by maximizing contribution to first node."""
double FBT::get_hu2p(func_t f , double q, double width){
    double Q = this->Q;

    // Create a lambda that captures the width, the func_t, and q, 
    // and will compute the function you want to find the minima of
    auto f2 = [f, width, q](double x) { return f_for_get_hu2p(x, width, f, q); };

    double zero1 = jn_zeros0[0];
    const int double_bits = std::numeric_limits<double>::digits;
    std::pair<double, double> r = boost::math::tools::brent_find_minima(f2, Q/10., 10.*Q, double_bits);

    double hu = r.first/zero1*M_PI;
    if(hu >= 2.){
        hu = 2.;
        //std::cerr<< "Warning: Number of nodes is too small N = " << this->N << std::endl;
    }

    return hu;
};



//"Determine transformed ht from untransformed hu."
double FBT::get_ht(double hu){
  int N = this->N;

  double zeroN = double(jn_zeros0[N-1]);

  return M_PI/zeroN*asinh(2./M_PI*atanh(hu/M_PI));
};

double FBT::fbt2p(double q, double width){

    double hu2p = 0.0;
    double ht2p = 0.0;
    int N = this->N;

    // Checking the number of CUDA devices and setting the current device.
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    if (num_devices > 1) {
        //std::cout << "Multiple GPUs detected. Selecting GPU 0." << std::endl;
        // Select GPU
        int device = 0;  
        cudaSetDevice(device);
    } else if (num_devices == 1) {
        //std::cout << "One GPU detected." << std::endl;
    } else {
        //std::cout << "No CUDA-capable GPU detected." << std::endl;
        return -1;  
    }

    // allocate GPU memory
    double* d_res;
    double* d_h;

    // Allocate device memory
    double* d_xi;
    double* d_w;

    cudaMalloc(&d_xi, xi.size() * sizeof(double));
    cudaMalloc(&d_w, w.size() * sizeof(double));

    cudaMemcpy(d_xi, xi.data(), xi.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w.data(), w.size() * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_res, sizeof(double));
    cudaMalloc((void**)&d_h, sizeof(double));

    double* d_block_results;  // Device array to hold per-block results
    cudaMalloc(&d_block_results, N * sizeof(double));

    // define block and grid size
    dim3 blockSize(128);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    cudaDeviceSynchronize();
    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d before launching kernel!\n", cudaStatus);
    }

    double* h_block_results = new double[N];
    double final_result;

    if (this->option == 0){ // default modified Ogata
        double hu2p = get_hu2p(func_ , q, width); //2d
        ht2p = get_ht(hu2p);  
        cudaMemcpy(d_h, &ht2p, sizeof(double), cudaMemcpyHostToDevice);
        ogatat_kernel2p<<<gridSize, blockSize>>>(N, q, width, d_h, d_xi, d_w, d_block_results);
        
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Error after kernel launch: %s\n", cudaGetErrorString(err));
        }

        //reducing the results
        cudaMemcpy(h_block_results, d_block_results, N * sizeof(double), cudaMemcpyDeviceToHost);
        final_result = std::accumulate(h_block_results, h_block_results + N, 0.0);
        } else if (this->option == 1){ // unmodified Ogata
        double hu2p = get_hu2p(func_ , q, width); //2d

        cudaMemcpy(d_h, &hu2p, sizeof(double), cudaMemcpyHostToDevice);                        
        ogatau_kernel2p<<<gridSize, blockSize>>>(N, q, width, d_h, d_xi, d_w, d_block_results);
        
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Error after kernel launch: %s\n", cudaGetErrorString(err));
        }

        //reducing the results
        cudaMemcpy(h_block_results, d_block_results, N * sizeof(double), cudaMemcpyDeviceToHost);
        final_result = std::accumulate(h_block_results, h_block_results + N, 0.0);

        } else if (this->option == 2){ // modified Ogata h = 0.05
        hu2p = 0.05;
        cudaMemcpy(d_h, &hu2p, sizeof(double), cudaMemcpyHostToDevice);
        ogatat_kernel2p<<<gridSize, blockSize>>>(N, q, width, d_h, d_xi, d_w, d_block_results);

        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Error after kernel launch: %s\n", cudaGetErrorString(err));
        }

        //reducing the results
        cudaMemcpy(h_block_results, d_block_results, N * sizeof(double), cudaMemcpyDeviceToHost);
        final_result = std::accumulate(h_block_results, h_block_results + N, 0.0);
    };

    // copy back result from device to host
    double result = final_result; 

    // Free device memory
    cudaFree(d_xi);
    cudaFree(d_w);
    cudaFree(d_h);
    cudaFree(d_block_results);

    delete[] h_block_results;

    return result;
}
