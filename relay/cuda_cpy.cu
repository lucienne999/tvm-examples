#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <tvm/runtime/ndarray.h>

__global__ void generate_random_kernel(float *data, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        // Feed a seed to curand
        curandState state;
        curand_init(clock() + index, 0, 0, &state);
        data[index] = curand_uniform(&state);
    }
}

void generateRandomData(float* devPtr, int N) {
    /* CUDA Kernel to generate random floats in device */
    int threads = 1024;
    int blocks = (N + threads - 1) / threads;
    generate_random_kernel<<<blocks, threads>>>(devPtr, N);

    // synchronize
    cudaDeviceSynchronize();
}

int main() {
    // Device context
    DLDevice dev{kDLCUDA, 0};

    // Data shape and type
    int batch_size = 1;
    int inc = 784;
    DLDataType dtype = {kDLFloat, 32, 1};

    // Allocate the DLTensor
    DLTensor* input = NULL;
    int64_t shape[2] = {batch_size, inc};
    int ndim = 2;
    TVMArrayAlloc(shape, ndim, dtype.code, dtype.bits, dtype.lanes, dev.device_type, dev.device_id, &input);

    if (input) {
        // Generate and assign the data to the GPU tensor
        generateRandomData(static_cast<float*>(input->data), batch_size * inc);

        float* hostData = new float[batch_size * inc];
        memset(hostData, 0, batch_size * inc * sizeof(float));
        // Copy data from device to host
        cudaMemcpy(hostData, input->data, batch_size * inc * sizeof(float), cudaMemcpyDeviceToHost);

        // Print the data
        for (int i = 0; i < batch_size * inc; ++i) {
            std::cout << hostData[i] << " ";
        }
        std::cout << std::endl;
        delete[] hostData;

    }

    cudaFree(input->data);
    TVMArrayFree(input);
}