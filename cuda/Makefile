NVCC = nvcc
NVCC_FLAGS =  -std=c++11 -arch=sm_50
G++ = g++

EXECUTABLE = fourier_bessel_transform
BOOST_EXECUTABLE = boost_math

SRC_DIR = src
INC_DIR = include

CUDA_SOURCES = $(SRC_DIR)/FBT.cu $(SRC_DIR)/example.cu
CPP_SOURCE = $(SRC_DIR)/BoostMath.cpp

all: $(EXECUTABLE) $(BOOST_EXECUTABLE)

$(EXECUTABLE): $(CUDA_SOURCES)
	$(NVCC) $(NVCC_FLAGS) -I$(INC_DIR) -o $@ $^

$(BOOST_EXECUTABLE): $(CPP_SOURCE)
	$(G++) -o $@ $^

clean:
	rm -f $(EXECUTABLE) $(BOOST_EXECUTABLE)