#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256

__global__ void matrix_unrolling_kernel(const float *input, float *output,
                                        const int Batch, const int Channel,
                                        const int Height, const int Width,
                                        const int K) {
    /*
    Modify this function to implement the input matrix unrolling kernel.

    Function paramter definitions:
    input - input
    output - output
    Batch - batch_size (number of images in x)
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    // const int unrolled_height = Channel * K * K;
    const size_t unrolled_width = Batch * Height_out * Width_out;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // __shared__ float shared_in[TILE_WIDTH + K - 1][TILE_WIDTH + K - 1];

    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define out_2d(i1, i2) output[(i1) * unrolled_width + (i2)]
    // TODO: Insert your input matrix unrolling kernel code here

    // int batch_idx = blockIdx.z;

    // Calculate the output position
    // int row_o = blockIdx.y * TILE_WIDTH + threadIdx.y;
    // int col_o = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int t = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int c = t/unrolled_width;
    int s = t % unrolled_width;
    int b = s/(Height_out * Width_out);
    int onebatch_idx = s % (Height_out * Width_out);
    int h_out = onebatch_idx / Width_out;
    int w_out = onebatch_idx % Width_out;
    size_t w_unroll = b * (Height_out * Width_out) + h_out * Width_out + w_out;
    int w_base = c * (K * K);
    // Check if within output bounds
    if ( t < Batch * Channel * Height_out * Width_out) {
        // Unroll the KxK region and concatenate channels along rows
            for (int p = 0; p < K; ++p) {
                for (int q = 0; q < K; ++q) {
                    size_t h_unroll = w_base + p * K + q;
                    out_2d(h_unroll, w_unroll) = in_4d(b, c, h_out + p, w_out + q);
                }
            }
        }
    

    #undef in_4d
    #undef out_2d
}

// Tiled matrix multiplication kernel. Computes C = AB
// You don't need to modify this kernel.
__global__ void matrixMultiplyShared(const float *A, const float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns)
{
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;

    int row = by * TILE_WIDTH + ty, col = bx * TILE_WIDTH + tx;
    float val = 0;

    for (int tileId = 0; tileId < (numAColumns - 1) / TILE_WIDTH + 1; tileId++) {
        if (row < numARows && tileId * TILE_WIDTH + tx < numAColumns) {
            tileA[ty][tx] = A[(size_t) row * numAColumns + tileId * TILE_WIDTH + tx];
        } else {
            tileA[ty][tx] = 0;
        }
        if (col < numBColumns && tileId * TILE_WIDTH + ty < numBRows) {
            tileB[ty][tx] = B[((size_t) tileId * TILE_WIDTH + ty) * numBColumns + col];
        } else {
            tileB[ty][tx] = 0;
        }
        __syncthreads();

        if (row < numCRows && col < numCColumns) {
            for (int i = 0; i < TILE_WIDTH; i++) {
                val += tileA[ty][i] * tileB[i][tx];
            }
        }
        __syncthreads();
    }

    if (row < numCRows && col < numCColumns) {
        C[row * numCColumns + col] = val;
    }
}

// Permutes the matmul result.
// The output feature map after matmul is of shape Map_out x Batch x Height_out x Width_out,
// and we need to permute it into Batch x Map_out x Height_out x Width_out.
// You don't need to modify this kernel.
__global__ void matrix_permute_kernel(const float *input, float *output, int Map_out,
                                      int Batch, int image_size) {
    int b = blockIdx.y;
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (x < image_size) {
        for (int m = 0; m < Map_out; m++) {
            output[b * Map_out * image_size + m * image_size + x] =
                    input[m * Batch * image_size + b * image_size + x];
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    //const to new pointers
    // float* ptr = (float*) host_output;
    // Allocate memory
    cudaMalloc((void**)device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    cudaMalloc((void**)device_output_ptr, Batch * Map_out * Height_out * Width_out * sizeof(float));
    cudaMalloc((void**)device_mask_ptr, Map_out * Channel * K * K * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);
    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

}

__global__ void fused_conv_kernel(const float *input, const float *mask, float *output,
                                  int Batch, int Map_out, int Channel,
                                  int Height, int Width, int K) {
    // Shared memory for tile-based matrix multiplication
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int out_image_size = Height_out * Width_out;
    const int Width_unrolled = Batch * out_image_size;
    const int Height_unrolled = Channel * K * K;

    // Global thread indices
    int by = blockIdx.y, bx = blockIdx.x;
    int ty = threadIdx.y, tx = threadIdx.x;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float result = 0.0;

    for (int tileId = 0; tileId < (Height_unrolled - 1) / TILE_WIDTH + 1; tileId++) {
        // Unrolling: Load tiles from input into shared memory
        if (row < Map_out && tileId * TILE_WIDTH + tx < Height_unrolled) {
            tileA[ty][tx] = mask[row * Height_unrolled + tileId * TILE_WIDTH + tx];
        } else {
            tileA[ty][tx] = 0.0;
        }

        if (col < Width_unrolled && tileId * TILE_WIDTH + ty < Height_unrolled) {
            // Unroll the input matrix on-the-fly
            int c = (tileId * TILE_WIDTH + ty) / (K * K);
            int s = (tileId * TILE_WIDTH + ty) % (K * K);
            int p = s / K;
            int q = s % K;

            int b = col / out_image_size;
            int onebatch_idx = col % out_image_size;
            int h_out = onebatch_idx / Width_out;
            int w_out = onebatch_idx % Width_out;

            if (b < Batch && c < Channel && h_out + p < Height && w_out + q < Width) {
                tileB[ty][tx] = input[b * Channel * Height * Width +
                                      c * Height * Width +
                                      (h_out + p) * Width +
                                      (w_out + q)];
            } else {
                tileB[ty][tx] = 0.0;
            }
        } else {
            tileB[ty][tx] = 0.0;
        }

        __syncthreads();

        // Matrix multiplication
        if (row < Map_out && col < Width_unrolled) {
            for (int i = 0; i < TILE_WIDTH; i++) {
                result += tileA[ty][i] * tileB[i][tx];
            }
        }

        __syncthreads();
    }

    // Permutation step
    if (row < Map_out && col < Width_unrolled) {
        int b = col / out_image_size;
        int onebatch_idx = col % out_image_size;
        int h_out = onebatch_idx / Width_out;
        int w_out = onebatch_idx % Width_out;

        if (b < Batch && h_out < Height_out && w_out < Width_out) {
            output[b * Map_out * out_image_size +
                   row * out_image_size +
                   onebatch_idx] = result;
        }
    }
}

// __host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     const int Height_out = Height - K + 1;
//     const int Width_out = Width - K + 1;
//     const int Height_unrolled = Channel * K * K;
//     const int Width_unrolled = Batch * Height_out * Width_out;

//     float *unrolled_matrix;  // Pointer to device memory for storing the unrolled matrix
//     float *matmul_output;    // Pointer to device memory for storing the result of matrix multiplication
//     cudaMalloc((void**)&unrolled_matrix, (size_t) Batch * Channel * K * K * Height_out * Width_out * sizeof(float));
//     cudaMalloc((void**)&matmul_output, (Batch * Map_out * Height_out * Width_out) * sizeof(float));

//     // TODO: Set the kernel dimensions and call the matrix unrolling kernel.
//     int num_threads = Batch * Channel * Height_out * Width_out;
//     int num_blocks = ceil((num_threads) / BLOCK_SIZE);
//     // dim3 GridDim1((Width_out + TILE_WIDTH - 1) / TILE_WIDTH, (Height_out + TILE_WIDTH - 1) / TILE_WIDTH, Batch);
//     dim3 GridDim1(num_blocks,1,1);
//     dim3 BlockDim1(BLOCK_SIZE,1,1);
//     matrix_unrolling_kernel<<<GridDim1, BlockDim1>>>(device_input, unrolled_matrix, Batch, Channel, Height, Width, K);

//     // TODO: Set the kernel dimensions and call the matmul kernel
//     dim3 GridDim2((Width_unrolled + TILE_WIDTH - 1) / TILE_WIDTH, (Map_out + TILE_WIDTH - 1) / TILE_WIDTH);
//     dim3 BlockDim2(TILE_WIDTH, TILE_WIDTH);

//     // Call the matrix multiplication kernel
//     matrixMultiplyShared<<<GridDim2, BlockDim2>>>(
//         device_mask, unrolled_matrix, matmul_output, Map_out, Height_unrolled, Height_unrolled, Width_unrolled, Map_out, Width_unrolled);   
//     // Permute the result of matrix multiplication
//     const int out_image_size = Height_out * Width_out;
//     dim3 permute_kernel_grid_dim((out_image_size - 1) / BLOCK_SIZE + 1, Batch, 1);
//     matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE>>>(
//         matmul_output, device_output, Map_out, Batch, out_image_size
//     );

//     cudaFree(matmul_output);
//     cudaFree(unrolled_matrix);
// }

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask,
                                             const int Batch, const int Map_out, const int Channel,
                                             const int Height, const int Width, const int K) {
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    // Define grid and block dimensions
    dim3 BlockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 GridDim((Batch * Height_out * Width_out + TILE_WIDTH - 1) / TILE_WIDTH,
                 (Map_out + TILE_WIDTH - 1) / TILE_WIDTH);

    // Launch fused kernel
    fused_conv_kernel<<<GridDim, BlockDim>>>(device_input, device_mask, device_output,
                                             Batch, Map_out, Channel, Height, Width, K);
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;    
    cudaMemcpy(host_output, device_output, Batch * Map_out * Height_out * Width_out * sizeof(float), cudaMemcpyDeviceToHost);
    // TODO: Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}