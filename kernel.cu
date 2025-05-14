
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

// Matrix multiplication is really just a bunch of dot products,
// so I think it is intuitive to design the kernel to be a dot product function
__global__ void dotProductKernel(int *A, int *B, int *C)
{ 
    // A and B are matrices (stored as arrays) to multiply, C is final result
    // This kernel will find dot product between row in A and col in B and store this in slot in C
    
}

int main()
{
    /*
    These are what the default matrices look like:

    A:
    1 2 3
    4 5 6
    
    B:
    10 11
    20 21
    30 31
    
    AB = C
    */

    // The dimensions of the matrices, change these if you are 
    // multiplying your own matrices
    const int A_numRows = 2;
    const int A_numCols = 3;
    const int B_numRows = 3;
    const int B_numCols = 2;

    int *A, * B, * C; // Host matrices
    int *dev_A, *dev_B, *dev_C; // Corresponding device matrices

    // Allocating the hsot memory
    A = (int*)malloc(A_numRows * A_numCols * sizeof(int));
    B = (int*)malloc(B_numRows * B_numCols * sizeof(int));

    // The output matrix dimensions will be A_numRows x B_numCols
    C = (int*)malloc(A_numRows * B_numCols * sizeof(int));


    // Now allocate the corresponding device memory
    cudaMalloc((void**)&dev_A, A_numRows * A_numCols * sizeof(int));
    cudaMalloc((void**)&dev_B, B_numRows * B_numCols * sizeof(int));
    cudaMalloc((void**)&dev_C, A_numRows * B_numCols * sizeof(int));

    // This next part is just populating the matrices A and B
    for (int i = 0; i < A_numRows * A_numCols; i++) {
        A[i] = i;
    }

    for (int i = 0; i < B_numRows * B_numCols; i++) {
        B[i] = i*2;
    }

    // TODO: Copy memory to device, run kernel, free memory

    return 0;
}