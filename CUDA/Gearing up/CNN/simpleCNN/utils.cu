%%writefile utils.cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// CUDA error check
static void checkCuda(cudaError_t err, const char *msg){

    if(err != cudaSuccess){ 
        fprintf(stderr,"CUDA error (%s): %s\n", msg, cudaGetErrorString(err)); 
        exit(1);
    }
}

// Random number helper
// frand returns a random float in the range [-1, 1]
float frand(){ 
    // rand returns an integer between 0 and RAND_MAX
    // RAND_MAX is a constant defined in <stdlib.h>, and it can vary by the system. (ex: 32767)
    // We are dividing by RAND_MAX so the resultant float can be in the range [0.0, 1.0]
    // Then we scale (multiplying by 2 => [0.0, 2.0]) and shift (subtracting 1 => [-1.0, 1.0])
    // Starting random weights in the [-1, 1] range helps balance activation early on, so that the network doesnt start biased
    // It is often better for initial symmetry breaking in NNs.
    return (((float)rand() / RAND_MAX) * 2) - 1; 
}

// Initialize weights and dividing them by 100 so that they stay small ("relatively small")
void init_weights(float* W,int n){ 
    for(int i = 0; i < n; i++) {
        W[i] = 0.01f * frand(); 
    }
}

// Zero out biases
void zero_fill(float* p,int n){ 
    for(int i = 0; i < n; i++) {
        p[i] = 0;
    }
}

// Softmax + cross entropy
void softmax_and_loss_host(const float* logits, const unsigned char* labels, int N, int O, float* out_probs, float* out_loss){
    float total_loss=0;
    for(int n = 0; n < N; n++){
        const float* row = logits + (n * O);
        float m = row[0]; // max for numeric stability
        
        for(int j = 1; j < O; j++) {
            if(row[j] > m){
                m = row[j];
            }
        }

        double sum=0;
        for(int j = 0; j < O; j++) {
            sum += exp((double)row[j] - m);
        } 

        for(int j = 0; j < O; j++) {
            out_probs[n * O +j] = (float)(exp((double)row[j] - m) / sum);
        }

        int t = labels[n];
        total_loss += -logf(out_probs[(n * O) + t] + 1e-12f); //Softmax activation
    }
    *out_loss = total_loss / (float)N;
}
