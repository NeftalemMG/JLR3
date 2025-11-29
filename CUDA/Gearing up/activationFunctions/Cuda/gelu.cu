%%writefile gelu.cu
#include <stdio.h>
#include <math.h>

__global__ void gelu(float *input, float *output, int N){
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    // This is the gelU logic:
    // The gelu is supposed to be a smoother version of ReLU.
    // As you might know, because all the values of the outputs less than 0 are 0, 
    // and all the values above it stay as they are, we have this graph
    // which suddenly has a steep slope to the positive y at the 0's position. 
    //                  /
    //                 / 
    //                / 
    //               /
    //              /
    //             /
    //            /
    // _________0/

    // The gelU smooths this out by basically scaling the numbers depending on how
    // how big/small of a number they are:
    // (a) Small negative numbers might become small negative output
    // (b) Small positive numbers are scaled slightly
    // (c) Big positives stay the same

    // This smooth cutoff often helps deep NNs train faster and more accurately.
        
    // This is the gelu approximation formula:
    // GELU(x) ≈ 0.5 * x * (1 + tanh(0.797885 * (x + 0.044715 * x^3)))

    if (threadId < N) {
        output[threadId] = 0.5f * input[threadId] * 
                           (1.0f + tanhf(0.797885f * 
                           (input[threadId] + 0.044715f * input[threadId] * input[threadId] * input[threadId])));
    }
}

int main() {

    float h_input[8] = {1, -2, 3, -4, 5000, -0.76, 332, 45};
    float h_output[8];
    float *d_input, *d_output;

    cudaMalloc(&d_input, sizeof(float) * 8);
    cudaMalloc(&d_output, sizeof(float) * 8);
    cudaMemcpy(d_input, h_input, sizeof(float) * 8, cudaMemcpyHostToDevice);
    gelu<<<1, 8>>>(d_input, d_output, 8);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, sizeof(float) * 8, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 8; i++){
        printf("%f\n", h_output[i]);
    }
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}