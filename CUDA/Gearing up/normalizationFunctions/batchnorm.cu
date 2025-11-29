%%writefile batchnorm.cu

#include <stdio.h>
#include <math.h>

__global__ void batchNorm(float *input, float *output, float mean, float variance, float gamma, float beta,  int N){
    // For this simpler implementation, 
    // the mean and the variance will be computed in the CPU, 
    // but for the CNN we are going to build later, we will move those operations to the GPU

    int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    // This is the implementation of the batch norm normalization function
    // This is the actual math formula:
    // X = (x − μ) / sqrt(σ2+ϵ)
    //  where:
    //    x: input value
    //    μ: mean
    //    σ2: variance
    //    ϵ: small constant to prevent division by zero (In our case, 1e-5)

    if (threadID < N){
        float normalizedOutput = (input[threadID] - mean) / sqrtf(variance + 1e-5);
        
        // Then we apply a learnable scale and shift:
        //  y = γX + β 
        //  Where:
        //    γ: scale parameter
        //    β: shift parameter
        //    X: The output of the normalized function

        output[threadID] = (gamma * normalizedOutput) + beta;                
    }
}

int main() {

    int n = 1000;
    size_t bytes = n * sizeof(float);

    // Little C recap:
    // size_t is unsigned, meaning it can only represent positive int values.
    // I then asked what the difference between unsigned int and size_t might be
    // because I forgot about that,
    // and the difference is that size_t is an unsigned int type but is not
    // bound by the four bytes of memory that unsigned int occupies. It is bound 
    // by the largest implementation the system can handle.
    // TLDR: size_t can hold bigger values than unsigned int. 

    float *h_input = new float[n];
    float *h_output = new float[n];

    for (int i = 0; i < n; i++) {
        // This is just some random repeating pattern
        h_input[i] = (i % 100) / 10.0f;
    }

    float mean = 0.0f;
    for (int j = 0; j < n; j++){
        mean = mean + h_input[j];
        mean = mean/n;
    }

    float variance = 0.0f;
    for (int z = 0; z < n; z++){
        variance = ((variance) + ((h_input[z] - mean) * (h_input[z] - mean)) / n);
    }

    float gamma = 1.0f;
    float beta = 0.0f;

    float *d_input;
    float *d_output;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / n; // My first thought was floor division and then add 1, but the formula we have here is more resource efficient because it can handle the exact division case
    batchNorm<<<blocks, threads>>>(d_input, d_output, mean, variance, gamma, beta, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    for (int xXx = 0; xXx < 1000; xXx++){
        printf("Input %f: normalized %f\n", h_input[xXx], h_output[xXx]);
    }
    
    cudaFree(d_input);
    cudaFree(d_output);

    return 27;
    // why so serious ;)
}