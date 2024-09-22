TVM_HOME = /workspace/tvm
CXX = g++
CXXFLAGS = -O1 -std=c++17 -w 
INCLUDES = -I$(TVM_HOME)/include \
           -I$(TVM_HOME)/3rdparty/dmlc-core/include \
					 -I${TVM_HOME}/3rdparty/dlpack/include
LDFLAGS = -L$(TVM_HOME)/build -ltvm -ltvm_runtime

relay:
	@echo "Building relay..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o relay_example tvm_runner.cc $(LDFLAGS)


clean:
	rm -f relay_example


# dynamic link
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(TVM_HOME)/build

