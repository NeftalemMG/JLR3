%%writefile relu.cu

#include <stdio.h>

__global__ void Relu(float *input, float *output, int N){
    int threadId = threadIdx.x + (blockIdx.x * blockDim.x);
    // This is the RelU logic, 
    // If you don't know, RelU basically converts all outputs less than 0 to 0.
    // But if the output is greater than 0, it stays how it is.
    if (threadId < N){
        if (input[threadId] > 0) {
            output[threadId] = input[threadId];
        }
        else{
            output[threadId] = 0;
        }
    }

}

int main (){
    float h_IntegerArray[8] = {1, -2, 3, -4, 5, -1, 7, -8};
    float h_Output[8];
    float *d_IntegerArray;

    cudaMalloc(&d_IntegerArray, 8 * sizeof(float));
    cudaMemcpy(d_IntegerArray, h_IntegerArray, sizeof(float) * 8, cudaMemcpyHostToDevice);
    Relu<<<1, 8>>>(d_IntegerArray, d_IntegerArray, 8);
    cudaDeviceSynchronize();

    cudaMemcpy(h_Output, d_IntegerArray, sizeof(float) * 8, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 8; i++){
        printf("%f\n", h_Output[i]);
    }
    cudaFree(d_IntegerArray);
    return 0;
}