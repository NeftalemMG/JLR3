%%writefile crossentropy.cu

#include <stdio.h>
#include <math.h>

__global__ void crossentropy(float *pred, float *target, float *loss, int n ){
    int threadID = threadIdx.x + (blockIdx.x * blockDim.x);
    // This is the implementation of the cross-entropy loss function
    // Cross-entropy is used for classification problems and measures how
    // far the predicted probabilities are from the true one-hot encoded targets
    // For one sample, the cross entropy (L) is calculated as
    // L=−(∑(i) (yi​log(pi)))
    // where
    // yi is 1 for the correct class, 0 otherwise
    // pi is the predicted probability for class i
    // That sounds pretty straight forward doesnt it?
    
    if (threadID < n){
        if (target[threadID] == 1.0f){
            loss[threadID] = -logf(pred[threadID] + 1e-9f); // We are adding this super small number to avoid log(0) which is undefined
        }
        else {
            loss[threadID] = 0.0f;
            // This else statement is if the target value is 0.
            // Ex: pred = [0.9, 0.1, 0.8, 0.2]
            //     target = [1, 0, 1, 0]
        }
    }
    
}

int main () {
    size_t bytes = sizeof(float) * 4;
    
    float h_pred[4] = {0.1, 0.7, 0.2, 0.0};
    float h_target[4] = {0.0, 1.0, 0.0, 0.0};

    float *h_loss = new float[4];
    float *d_loss;
    cudaMalloc(&d_loss, bytes);
                
    float *d_pred;
    float *d_target;
    cudaMalloc(&d_pred, bytes);
    cudaMalloc(&d_target, bytes);
    
    cudaMemcpy(d_pred, h_pred, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, h_target, bytes, cudaMemcpyHostToDevice);
    
    int threads = 4;
    int blocks = 1;

    crossentropy<<<blocks, threads>>>(d_pred, d_target, d_loss, 4);
    cudaDeviceSynchronize();
    cudaMemcpy(h_loss, d_loss, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; i++){
        printf("Loss[%d] = %f\n", i, h_loss[i]);
    }
    
    cudaFree(d_pred);
    cudaFree(d_target);
    cudaFree(d_loss);

    return 27;
}