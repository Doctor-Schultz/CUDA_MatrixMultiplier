
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <chrono>

// cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

// Matrix multiplication is really just a bunch of dot products,
// so I think it is intuitive to design the kernel to be a dot product function
__global__ void dotProductKernel(int *A, int *B, int *C, int A_numRows, int A_numCols, int B_numRows, int B_numCols)
{ 
    // Each block uses the same col in B, and each thread in said block looks at a different row in A
    
    int B_column = blockIdx.x;
    int A_row = threadIdx.x;

    // A and B are matrices (stored as arrays) to multiply, C is final result
    // This kernel will find dot product between row in A and col in B and store this in slot in C

    int productSum = 0; // Represents the final dot product value, which is the sum of products of values in A and B
    
    // int indexToPrint = (rowIndex * numCols) + colIndex;
    // The outer loop is "moving down" the column in B, and at each point multiplies that col value by its corresponding A row value
    for (int B_colDepth = 0; B_colDepth < B_numRows; B_colDepth++) {
        // The indexing is best explained with this image: https://algebra2coach.com/wp-content/uploads/2021/08/how-to-do-matrix-multiplication_5.png
        int B_colValueIndex = (B_colDepth * B_numCols) + B_column;
        int B_colValue = B[B_colValueIndex];

        int A_rowValueIndex = (A_row * A_numCols) + B_colDepth;
        int A_rowValue = A[A_rowValueIndex];

        productSum += B_colValue * A_rowValue;
    }

    
    // Store the resulting answer in the C matrix

    // Answer stored in B_column, A_row
    // The output matrix dimensions will be A_numRows x B_numCols
    // int indexToPrint = (rowIndex * numCols) + colIndex;
    
    // Wrapper ints that make indexing C more intuitive
    int C_numCols = B_numCols;
    int C_numRows = A_numRows;
    int C_resultIndex = (A_row * C_numCols) + B_column;

    C[C_resultIndex] = productSum;

}

void printMatrix(int* matrix, int numRows, int numCols) {

    for (int rowIndex = 0; rowIndex < numRows; rowIndex++) {
        
        for (int colIndex = 0; colIndex < numCols; colIndex++) {
            
            int indexToPrint = (rowIndex * numCols) + colIndex;

            // Use a single space or double space after number depending on if it is double digit,
            // so that columns line up with each other
            if (matrix[indexToPrint] < 10) {
                printf("%d  ", matrix[indexToPrint]);
            }
            else {
                printf("%d ", matrix[indexToPrint]);
            }
        }
        printf("\n");
    }
}

void matrixMultiplyCPU(const int* A, const int* B, int* C, int A_numRows, int A_numCols, int B_numRows, int B_numCols) {
    for (int row = 0; row < A_numRows; row++) {
        for (int col = 0; col < B_numCols; col++) {
            int sum = 0;
            for (int k = 0; k < A_numCols; k++) {
                sum += A[(row * A_numCols) + k] * B[(k * B_numCols) + col];
            }
            C[(row * B_numCols) + col] = sum;
        }
    }
}

int main()
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
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
    const int A_numRows = 5;
    const int A_numCols = 10;
    const int B_numRows = 10;
    const int B_numCols = 3;

    int *A, * B, * C, *cpu_C; // Host matrices
    int *dev_A, *dev_B, *dev_C; // Corresponding device matrices

    // Allocating the hsot memory
    A = (int*)malloc(A_numRows * A_numCols * sizeof(int));
    B = (int*)malloc(B_numRows * B_numCols * sizeof(int));

    // The output matrix dimensions will be A_numRows x B_numCols
    C = (int*)malloc(A_numRows * B_numCols * sizeof(int));
    cpu_C = (int*)malloc(A_numRows * B_numCols * sizeof(int));


    // Now allocate the corresponding device memory
    cudaMalloc((void**)&dev_A, A_numRows * A_numCols * sizeof(int));
    cudaMalloc((void**)&dev_B, B_numRows * B_numCols * sizeof(int));
    cudaMalloc((void**)&dev_C, A_numRows * B_numCols * sizeof(int));

    // This next part is just populating the matrices A and B
    for (int i = 0; i < A_numRows * A_numCols; i++) {
        A[i] = i + (i*3);
    }

    for (int i = 0; i < B_numRows * B_numCols; i++) {
        B[i] = (i*2) + 19;
    }

    // Copy the A and B matrices to the device
    cudaMemcpy(dev_A, A, A_numRows * A_numCols * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, B_numRows * B_numCols * sizeof(int), cudaMemcpyHostToDevice);

    
    // Refuse to multiply if the dimensions don't match
    if (A_numCols == B_numRows) {
        cudaEventRecord(start);
        dotProductKernel<<<B_numCols, A_numRows>>>(dev_A, dev_B, dev_C, A_numRows, A_numCols, B_numRows, B_numCols);
        cudaEventRecord(stop);
    }
    else {
        printf("ERROR: DIMENSIONS OF MATRICES A AND B DO NOT MATCH");
        
        cudaFree(dev_A);
        cudaFree(dev_B);
        cudaFree(dev_C);

        free(A);
        free(B);
        free(C);
        //free(cpu_C);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        return 0;
    }
    // TODO: call kernel here

    // Copy the resulting C matrix back to the host
    cudaMemcpy(C, dev_C, A_numRows * B_numCols * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU Kernel elapsed time: %f ms\n", milliseconds);

    // The following is just to time the sequential CPU version of matrix multiplicationg, to see how much faster my version is
    /*
    auto cpu_start = std::chrono::high_resolution_clock::now();
    matrixMultiplyCPU(A, B, cpu_C, A_numRows, A_numCols, B_numRows, B_numCols);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
    double cpu_time = cpu_duration.count() / 1000.0; // Convert to milliseconds
    printf("CPU Matrix Multiplication Time: %.3f ms\n",cpu_time);
    */

    
    // Print A, B, and C to the console
    printf("Matrix A:\n");
    printMatrix(A, A_numRows, A_numCols);
    printf("\nMatrix B:\n");
    printMatrix(B, B_numRows, B_numCols);
    printf("\nMatrix C:\n");
    printMatrix(C, A_numRows, B_numCols);
    
    // Free device and host memory
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    free(A);
    free(B);
    free(C);
    //free(cpu_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}