%%writefile swish.cu
#include <stdio.h>
#include <math.h>

__global__ void swish(float *input, float *output, int N){
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;

    // This is the cuda implementation of the swish activation function. 
    // Swish(x) = x * sigmoid(x)
    // Sigmoid(x) = 1 / (1 + (e^-x))
    // Swish graphs are kinda similar to gelu graphs
    // It is another smoother, non-monotonic version of relu; it allows smaller negative values
    // to pass through instead of cutting them off entirely.
    // One question that I had at first was that swish and gelu are kinda similar, 
    // why do we need one over the other?
    // Here was what our dev God (chatGPT) responded:
    
    // Use case	Effect:
    // Swish | Tends to keep more negative signals alive → smoother gradients.
    // GELU  |	More selective (kills small negatives) → better for deep Transformers (like BERT, GPT).
    
    // Moreover, Swish is slightly faster to compute on hardware, 
    // but GELU tends to perform slightly better in large models.

    // And for now, we will just leave it there. 
    if (threadId < N){
        float x = input[threadId];
        float sigmoid = (1.0f / (1.0f + expf(-x)));
        output[threadId] = x * sigmoid;
    }
}

int main () {
    float h_input[8] = {1, -2, -4, 5, 12304, 1234, -3343, 2223};
    float h_output[8];

    float *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(float) * 8);
    cudaMalloc(&d_output, sizeof(float) * 8);

    cudaMemcpy(d_input, h_input, sizeof(float) * 8, cudaMemcpyHostToDevice);
    swish<<<1, 8>>>(d_input, d_output, 8);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, sizeof(float) * 8, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 8; i++){
        printf("%f\n", h_output[i]);
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 7;
}

