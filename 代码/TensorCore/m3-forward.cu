#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <mma.h> 
#define __CUDA_ARCH__ 1180

#define BLOCK_TENSOR_X 32
#define BLOCK_TENSOR_Y 32
#define TILE_WIDTH 16
#define BLOCK_SIZE 256
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_SIZE 32

const int warpNum = BLOCK_TENSOR_X * BLOCK_TENSOR_Y / WARP_SIZE;
const int warpY = 8;
const int warpX = warpNum / warpY;

using namespace nvcuda;

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

    int batch_idx = blockIdx.z;

    // Calculate the output position
    int row_o = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col_o = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // Check if within output bounds
    if (row_o < Height_out && col_o < Width_out) {
        // Calculate the flat index for spatial position in output's width
        size_t w_o = row_o * Width_out + col_o + batch_idx * Height_out * Width_out;
        size_t h_o;
        // Unroll the KxK region and concatenate channels along rows
        for (int c = 0; c < Channel; ++c) {
            int idx = 0;
            for (int i = 0; i < K; ++i) {
                for (int j = 0; j < K; ++j) {
                    int row_in = row_o + i;
                    int col_in = col_o + j;
                    h_o = c * K * K + idx;
                    out_2d(h_o, w_o) = in_4d(batch_idx, c, row_in, col_in);
                    idx++;
                }
            }
        }
    }

    #undef in_4d
    #undef out_2d
}


// Tiled matrix multiplication kernel. Computes C = AB
// You don't need to modify this kernel.
__global__ void matrixMultiplyShared(const half *A, const half *B, float *C,
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



__global__ void matrixMultiplyTensorCore(half *dA, half *dB, float *dC, size_t M, size_t K, size_t N)
{
    int lda = K;
    int ldb = N;
    int ldc = N;

    // size_t indA = blockIdx.x * warpX * WMMA_M;
    // size_t indB = blockIdx.y * warpY * WMMA_N;

    int warpId = (threadIdx.x + threadIdx.y * blockDim.x) / WARP_SIZE;
    int warpIdx = warpId % warpX;
    int warpIdy = warpId / warpX;
    // size_t warp_M = (blockIdx.x * blockDim.x + threadIdx.x)/32;
    // size_t warp_N = (blockIdx.y * blockIdx.y) + threadIdx.y;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,half, wmma::row_major> frag_A;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_B;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_C;

    wmma::fill_fragment(frag_C, 0.0f);
    size_t aRow = blockIdx.y * warpY * WMMA_M + warpIdy * WMMA_M;
    size_t bCol = blockIdx.x * warpX * WMMA_N + warpIdx * WMMA_N;
    // size_t aRow = warp_M * WMMA_M;
    // size_t bCol = warp_N * WMMA_N;
    int width = (K + WMMA_K - 1) / WMMA_K;
    for (int i = 0; i < width; i++)
    {
        int aCol = i * WMMA_K;
        size_t bRow = i * WMMA_K;
        if (aRow < M && aCol < K && bRow < K && bCol < N)
        {
            wmma::load_matrix_sync(frag_A, dA + (size_t)(aRow * lda) + (size_t)aCol, lda);
            wmma::load_matrix_sync(frag_B, dB + (size_t)(bRow * ldb) + (size_t)bCol, ldb);
            wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);
        }
    }
    size_t cRow = aRow;
    size_t cCol = bCol;
    if (cRow < M && cCol < N)
    {
        wmma::store_matrix_sync(dC + (size_t)(cRow * ldc) + (size_t)cCol, frag_C, ldc, wmma::mem_row_major);
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

}

//helper function to pad matrices and convert the datatype to half
__global__ void padMatrixToHalfKernel(const float* input, __half* output, size_t rows, size_t cols, size_t padded_rows, size_t padded_cols) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    size_t y = blockIdx.y * blockDim.y + threadIdx.y; // Row index

    if (x < padded_cols && y < padded_rows) {
        if (x < cols && y < rows) {
            // Copy and convert float to half
            output[y * padded_cols + x] = __float2half(input[y * cols + x]);
        } else {
            // Fill padding with 0 in half precision
            output[y * padded_cols + x] = __float2half(0.0f);
        }
    }
}

__global__ void removePadding(float *paddedMatrix, float *originalMatrix, 
                              size_t paddedRows, size_t paddedCols, 
                              size_t originalRows, size_t originalCols) {
    // Calculate the row and column index in the original matrix
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the current thread corresponds to a valid position in the original matrix
    if (row < originalRows && col < originalCols) {
        // Map the element from the padded matrix to the original matrix
        originalMatrix[row * originalCols + col] = paddedMatrix[row * paddedCols + col];
    }
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, 
                                             const int Batch, const int Map_out, const int Channel, 
                                             const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const size_t Height_unrolled = Channel * K * K;
    const size_t Width_unrolled = Batch * Height_out * Width_out;

    float *unrolled_matrix;  // Pointer to device memory for unrolled matrix
    float *matmul_output;    // Pointer to device memory for result
    cudaMalloc(&unrolled_matrix, (size_t)Batch * Channel * K * K * Height_out * Width_out * sizeof(float));
    cudaMalloc(&matmul_output, Batch * Map_out * Height_out * Width_out * sizeof(float));

    // Unroll input
    dim3 GridDim1((Width_out + TILE_WIDTH - 1) / TILE_WIDTH, (Height_out + TILE_WIDTH - 1) / TILE_WIDTH, Batch);
    dim3 BlockDim1(TILE_WIDTH, TILE_WIDTH);
    matrix_unrolling_kernel<<<GridDim1, BlockDim1>>>(device_input, unrolled_matrix, Batch, Channel, Height, Width, K);

    // Allocate padded matrices
    size_t paddedARows = (Map_out + TILE_WIDTH - 1) / TILE_WIDTH * TILE_WIDTH;
    size_t paddedACols = (Height_unrolled + TILE_WIDTH - 1) / TILE_WIDTH * TILE_WIDTH;
    size_t paddedBCols = (Width_unrolled + TILE_WIDTH - 1) / TILE_WIDTH * TILE_WIDTH;

    half *device_mask_padded, *unrolled_matrix_padded;
    cudaMalloc(&unrolled_matrix_padded, (size_t)paddedACols * paddedBCols * sizeof(half));
    cudaMalloc(&device_mask_padded, (size_t)paddedARows * paddedACols * sizeof(half));

    // Pad device mask
    dim3 blockDim2(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim2((paddedACols + TILE_WIDTH - 1) / TILE_WIDTH, (paddedARows + TILE_WIDTH - 1) / TILE_WIDTH);
    padMatrixToHalfKernel<<<gridDim2, blockDim2>>>(device_mask, device_mask_padded, Map_out, Height_unrolled, paddedARows, paddedACols);
    cudaDeviceSynchronize();

    // Pad unrolled matrix
    dim3 blockDim3(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim3((paddedBCols + TILE_WIDTH - 1) / TILE_WIDTH, (paddedACols + TILE_WIDTH - 1) / TILE_WIDTH);
    padMatrixToHalfKernel<<<gridDim3, blockDim3>>>(unrolled_matrix, unrolled_matrix_padded, Height_unrolled, Width_unrolled, paddedACols, paddedBCols);
    cudaDeviceSynchronize();

    float* matmul_output_paddded;
    cudaMalloc(&matmul_output_paddded, (size_t)paddedARows * paddedBCols * sizeof(float));
    // Tensor Core Matrix Multiplication
    dim3 blockDim4(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim4((paddedBCols + TILE_WIDTH - 1) / TILE_WIDTH, (paddedARows + TILE_WIDTH - 1) / TILE_WIDTH);   
    // matrixMultiplyTensorCore<<<gridDim4, blockDim4>>>(device_mask_padded, unrolled_matrix_padded, matmul_output_paddded,
    //                                                   paddedARows, paddedACols, paddedACols, paddedBCols, paddedARows, paddedBCols);
    // matrixMultiplyShared<<<gridDim4, blockDim4>>>(device_mask_padded, unrolled_matrix_padded, matmul_output_paddded,
    //                                             paddedARows, paddedACols, paddedACols, paddedBCols, paddedARows, paddedBCols);
// Configuration for WMMA
    // int num_block_x = (paddedARows + WMMA_M * warpX - 1) / (WMMA_M * warpX);
    // int num_block_y = (paddedBCols + WMMA_N * warpY - 1) / (WMMA_N * warpY);
    int num_block_x = (paddedBCols + WMMA_N * warpX - 1) / (WMMA_N * warpX);
    int num_block_y = (paddedARows + WMMA_M * warpY - 1) / (WMMA_M * warpY);

    dim3 block_dim(BLOCK_TENSOR_X, BLOCK_TENSOR_Y, 1);
    dim3 grid_dim(num_block_x, num_block_y, 1);
    // dim3 blockSize(32, 32, 1);
    // int block_num_x = (paddedARows / 16 + blockSize.x - 1) / blockSize.x;
    // int block_num_y = (paddedBCols / 16 + blockSize.y - 1) / blockSize.y;
    // dim3 gridSize(block_num_x, block_num_y, 1);
    matrixMultiplyTensorCore<<<grid_dim, block_dim>>>(device_mask_padded, unrolled_matrix_padded, matmul_output_paddded, paddedARows, paddedACols, paddedBCols);

    //remove padding drom padded matmult output
    dim3 blockDim(16, 16); 
    dim3 gridDim((Width_unrolled + blockDim.x - 1) / blockDim.x, 
                (Map_out + blockDim.y - 1) / blockDim.y);
    removePadding<<<gridDim, blockDim>>>(matmul_output_paddded, matmul_output, 
                                        paddedARows, paddedBCols, 
                                        Map_out, Width_unrolled);
    cudaDeviceSynchronize();

    // Permute the result back
    const int out_image_size = Height_out * Width_out;
    dim3 permute_kernel_grid_dim((out_image_size - 1) / BLOCK_SIZE + 1, Batch, 1);
    matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE>>>(matmul_output, device_output, Map_out, Batch, out_image_size);
    cudaDeviceSynchronize();

    // Free device memory
    cudaFree(device_mask_padded);
    cudaFree(unrolled_matrix_padded);
    cudaFree(matmul_output);
    cudaFree(unrolled_matrix);
    cudaFree(matmul_output_paddded);
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