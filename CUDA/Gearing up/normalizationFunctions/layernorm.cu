%%writefile layernorm.cu

#include <stdio.h>
#include <math.h>

__global__ void layernorm(float *input, float *output, float gamma, float beta, int featuresPerSample){

    int sampleID = blockIdx.x; // Here, every block can handle each individual sample
    
    int offset = sampleID * featuresPerSample;
    // This was a little confusing to me at first, but if we have an example where this is the input:
    // Input = [
    //    [x11, x12, x13, x14],   // Sample 0 (F = 4 features)
    //    [x21, x22, x23, x24],   // Sample 1
    //    [x31, x32, x33, x34]    // Sample 2
    // ]
    // When we flatten this input into a 1D array for GPU memory, it becomes:
    // input = [x11, x12, x13, x14, x21, x22, x23, x24, x31, x32, x33, x34]
    // So that is why we need an offset to tell us where each sample starts in that flat array
    // and based on that, we are going to normalize one sample at a time (ie layer normalization).
    
    float mean = 0.0f;
    for (int i = 0; i < featuresPerSample; i++){
        mean += input[offset + i];
    }
    mean = mean / featuresPerSample; // Calculating the mean per sample
    
    float variance = 0.0f;
    // Little Recap: Variance is a measure of how spread out the values are around the mean
    for (int j = 0; j < featuresPerSample; j++){
        variance += ((input[offset + j] - mean) * (input[offset + j] - mean));
    }
    variance = variance / featuresPerSample; // Calculating the variance of the sample

    // Normalization - It is basically the same calculation that we saw previously in batch norm
    // You can refer to the batchnorm.cu file in the same dir, to see the formula. I dont want to write it again here. 
    for (int k = 0; k < featuresPerSample; k++) {
        float x_hat = (input[offset + k] - mean) / sqrtf(variance + 1e-5f);
        // x_hat is just a standard symbol in normalization equations. It just means the normalized version of x. 
        output[offset + k] = (gamma * x_hat) + beta;
    }   
}

int main () {
    int samples = 6;
    int featuresPerSample = 3;
    int total = samples * featuresPerSample;

    size_t bytes = total * sizeof(float);

    float *h_input = new float[total];
    float *h_output = new float[total];

    float sampleData[18] = {2.0, 4.0, 6.0, 1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 4.0, 4.0, 4.0, 3.5, 4.5, 56.6, 27.0, 27.6, 12.0};
    for (int i = 0; i < total; i++){
        h_input[i] = sampleData[i];
    }

    float gamma = 1.0f;
    float beta = 0.0f;

    float *d_input;
    float *d_output;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    int threads = 1; // We only need 1 thread per block since we loop inside
    int blocks = samples;
    layernorm<<<blocks, threads>>>(d_input, d_output, gamma, beta, featuresPerSample);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    printf("Layer Normalization Outputs\n");
    for (int xxx = 0; xxx < samples; xxx++){
        printf("Sample %d:\n", xxx);
        for (int zzz = 0; zzz < featuresPerSample; zzz++){
            int sampleID = (xxx * featuresPerSample) + zzz;
            printf("Input: %6.2f    normalized: %8.5f\n", h_input[sampleID], h_output[sampleID]);
        }
        printf("\n");
    }
   
    cudaFree(d_input);
    cudaFree(d_output);
    return 27;
}