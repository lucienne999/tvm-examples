TVM_HOME = /workspace/tvm
CXX = g++
CXXFLAGS = -O1 -std=c++17 -w 
INCLUDES = -I$(TVM_HOME)/include \
           -I$(TVM_HOME)/3rdparty/dmlc-core/include \
					 -I${TVM_HOME}/3rdparty/dlpack/include \
					 -I/usr/local/cuda/include
LDFLAGS = -L$(TVM_HOME)/build -ltvm -ltvm_runtime -L/usr/local/cuda/lib64 -lcudart -lcurand


relay_cpu:
	@echo "Building relay cpu..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o relay_example tvm_runner.cc $(LDFLAGS)

relay_cuda:
	@echo "Building relay gpu..."
	nvcc $(CXXFLAGS) $(INCLUDES) -o relay_cuda_example tvm_runner_cuda.cu $(LDFLAGS)


clean:
	rm -f relay_example relay_cuda_example
	rm -f cuda_cpy

test:
	nvcc $(CXXFLAGS) $(INCLUDES) -o cuda_cpy cuda_cpy.cu $(LDFLAGS) 

# dynamic link
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(TVM_HOME)/build
# export PATH=/usr/local/cuda/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH