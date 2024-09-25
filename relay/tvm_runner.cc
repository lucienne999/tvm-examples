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

// header, just put together for convenience
namespace TVMF {
class Runner {
public:
  Runner(const std::string& model_path) {
    // Load the model
    tvm::runtime::Module mod = tvm::runtime::Module::LoadFromFile(model_path);
    // Get the main function
    DLDevice dev = DLDevice{kDLCPU, 0};
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


DLTensor* Alloc(int64_t* shape, int nDim, DLDataType dtype, DLDevice dev) {

  int64_t total_size = 1;
  for (int i = 0; i < nDim; i++) {
    total_size *= shape[i];
  }

  size_t element_size = (dtype.bits / 8) * (dtype.lanes > 1 ? dtype.lanes : 1); // size of each element. if make sure is float, use sizeof(float) is fine
  std::cout << "[INFO] Allocate memory for tensor, total size: " << total_size << " element_size: " <<  element_size << std::endl;
  void* data = malloc(total_size * element_size);
  if (!data) {
    std::cerr << "Failed to allocate memory for tensor" << std::endl;
    return nullptr;
  }

  DLTensor* tensor = new DLTensor;
  tensor->data = data;
  tensor->ndim = nDim;
  tensor->dtype = dtype;
  tensor->device = dev;
  tensor->shape = new int64_t[nDim];
  std::copy(shape, shape + nDim, tensor->shape);
  return tensor;
}


void Free(DLTensor* tensor) {
  // free(tensor->data);
  // delete[] tensor->shape;
  delete tensor;
}


} // namespace TVMF


int main() {
  TVMF::Runner model("./mlp1.so");
  std::cout << "[INFO] Load model" << std::endl;
  // define input struct
  int batch_size = 1;
  int inc = 784;

  // allocate
  // 自己定义的 alloc 在 tvm 连续内存检查报错：不要自己写好了
  // InternalError: Check failed: (IsContiguous(*from) && IsContiguous(*to)) is false: CopyDataFromTo only support contiguous array for now
  // DLTensor* input = TVMF::Alloc(new int64_t[2]{batch_size, inc}, 2, DLDataType{kDLFloat, 32, 1}, DLDevice{kDLCPU, 0});

  DLTensor* input;
  int64_t shape[2] = {batch_size, inc};
  int ndim = 2;
  int dtype_code = kDLFloat;
  int dtype_bits = 32;
  int device_type = kDLCPU;
  int device_id = 0;
  int dtype_lanes = 1;

  // 使用 NDArray 可不操心内存
  // 可参考：https://github.com/apache/tvm/blob/main/apps/howto_deploy/cpp_deploy.cc#L86
  
  TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input);
 
  if (input) {
    float *data = static_cast<float*>(input->data);
    for (int i = 0; i < batch_size * inc; i++) {
      data[i] = static_cast<float>(rand() / RAND_MAX);
    }
  }
  std::cout << "set input data ..." << std::endl;

  model.set_input("inp0", input); // name should be same with export
  std::cout << "infer..." << std::endl;
  model.run();
  std::cout << "infer done" << std::endl;

  DLTensor* output; // dont need alloc for output
  int64_t out_shape[2] = {batch_size, 10};
  TVMArrayAlloc(out_shape, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &output);

  model.get_output(0, output);
  // debug info, print output shape and data
  std::cout << "Output Shape: ";
  for (int i = 0; i < output->ndim; ++i) {
    std::cout << output->shape[i] << " ";
  }
  float* output_data = static_cast<float*>(output->data);
  for (int i = 0; i < output->shape[1]; ++i) { // Adjust based on output shape
    std::cout << output_data[i] << " ";
  }
  std::cout << std::endl;

  TVMArrayFree(input);
  TVMArrayFree(output);
  return 0;
  
  
}