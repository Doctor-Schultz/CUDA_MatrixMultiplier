
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

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

    // Copy the A and B matrices to the device
    cudaMemcpy(dev_A, A, A_numRows * A_numCols * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, B_numRows * B_numCols * sizeof(int), cudaMemcpyHostToDevice);

    // Refuse to multiply if the dimensions don't match
    if (A_numCols == B_numRows) {
        dotProductKernel<<<B_numCols, A_numRows>>>(dev_A, dev_B, dev_C, A_numRows, A_numCols, B_numRows, B_numCols);
    }
    else {
        printf("ERROR: DIMENSIONS OF MATRICES A AND B DO NOT MATCH");
        
        cudaFree(dev_A);
        cudaFree(dev_B);
        cudaFree(dev_C);

        free(A);
        free(B);
        free(C);

        return 0;
    }
    // TODO: call kernel here

    // Copy the resulting C matrix back to the host
    cudaMemcpy(C, dev_C, A_numRows * B_numCols * sizeof(int), cudaMemcpyDeviceToHost);


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

    return 0;
}