// 偷懒复制粘贴版
#include <iostream>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/c_runtime_api.h>
#include <algorithm>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/container/adt.h>
#include <tvm/runtime/vm/executable.h>
#include <tvm/runtime/vm/vm.h>
#include <tvm/runtime/memory/memory_manager.h>
#include <dlpack/dlpack.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <tvm/runtime/ndarray.h>

// header, just put together for convenience
namespace TVMF {
class Runner {
public:
  Runner(const std::string& model_path) {
    // Load the model
    tvm::runtime::Module mod = tvm::runtime::Module::LoadFromFile(model_path);
    // Get the main function
    DLDevice dev = DLDevice{kDLCUDA, 0}; // Use GPU
    mModule = mod.GetFunction("default")(dev);
    mMainFunc = mModule.GetFunction("run");
    mInputFunc = mModule.GetFunction("set_input");
    mOutputFunc = mModule.GetFunction("get_output");
  }

  void set_input(const std::string& name, DLTensor* input) {
    mInputFunc(name, input);
  }

  void run() {
    mMainFunc();
  }

  void get_output(int index, DLTensor* output) {
    mOutputFunc(index, output);
  }

private:
  tvm::runtime::Module mModule;
  tvm::runtime::PackedFunc mMainFunc;
  tvm::runtime::PackedFunc mInputFunc;
  tvm::runtime::PackedFunc mOutputFunc;
};

__global__ void generate_random_kernel(float *data, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        // Feed a seed to curand
        curandState state;
        curand_init(clock() + index, 0, 0, &state);
        data[index] = curand_uniform(&state);
    }
};

void generateRandomData(float* devPtr, int N) {
    /* CUDA Kernel to generate random floats in device */
    int threads = 1024;
    int blocks = (N + threads - 1) / threads;
    generate_random_kernel<<<blocks, threads>>>(devPtr, N);
    // synchronize
    cudaDeviceSynchronize();
};

void DLManagedTensorDeleter(DLManagedTensor* self) {
    if (self) {
        TVMArrayFree(&self->dl_tensor);
        delete self;
    }
};
// Convert DLTensor* to DLManagedTensor*
DLManagedTensor* ToDLManagedTensor(DLTensor* dl_tensor) {
    DLManagedTensor* managed_tensor = new DLManagedTensor;
    if (!managed_tensor) {
        throw std::runtime_error("Could not allocate DLManagedTensor");
    }

    managed_tensor->dl_tensor = *dl_tensor;
    managed_tensor->manager_ctx = nullptr;
    managed_tensor->deleter = DLManagedTensorDeleter;

    return managed_tensor;
}
} // namespace TVMF

int main() {
  TVMF::Runner model("./mlp1_cuda.so");
  DLDevice dev = DLDevice{kDLCUDA, 0}; 
  std::cout << "[INFO] Load model" << std::endl;
  // define input struct
  int batch_size = 1;
  int inc = 784;
  
  DLTensor* input;
  int64_t shape[2] = {batch_size, inc};
  int ndim = 2;
  int dtype_code = kDLFloat;
  int dtype_bits = 32;
  // Change device type to kDLCUDA
  int device_type = kDLCUDA; // Use GPU
  int device_id = 0;
  int dtype_lanes = 1;
  // locate GPU
  TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input);
  // gen data in GPU
  TVMF::generateRandomData(static_cast<float*>(input->data), batch_size * inc);

  model.set_input("inp0", input); 
  std::cout << "infer..." << std::endl;
  model.run();
  std::cout << "infer done" << std::endl;

  DLTensor* output; 
  int64_t out_shape[2] = {batch_size, 10};
  TVMArrayAlloc(out_shape, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &output);

  model.get_output(0, output);
  // output is CUDA DLTensor, use NDArray to fetch
  DLManagedTensor* managed_output = TVMF::ToDLManagedTensor(output);
  tvm::runtime::NDArray output_nd = tvm::runtime::NDArray::FromDLPack(managed_output);
  tvm::runtime::NDArray output_nd_cpu = output_nd.CopyTo(DLDevice{kDLCPU, 0});
  float* host_data = static_cast<float*>(output_nd_cpu->data);
  for (int i = 0; i < 10; i++) { // Print first 10 values as a check
    std::cout << host_data[i] << " ";
  }
  std::cout << std::endl;

  // or use cudaMemcpy
  float* host_data2 = new float[batch_size * 10];
  cudaMemcpy(host_data2, output->data, batch_size * 10 * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 10; i++) {
    std::cout << host_data2[i] << " ";
  }
  std::cout << std::endl;


  TVMArrayFree(input);
  TVMArrayFree(output);
  return 0;
}