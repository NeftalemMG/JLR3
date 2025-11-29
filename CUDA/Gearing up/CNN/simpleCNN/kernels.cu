%%writefile kernels.cu

#include <cuda_runtime.h>
#include <stdio.h>

// This is a very simple conv2d without optimizations
// At its core, convolution is basically sliding small templates(filter/kernel) over data and compute weighted sums
// to see how well the data matches the template. 2D convolution means taking an 2D array

// In CNNs channels basically signify the depth of the data, 
// One can think of channeels as separate layers stacked on top of each other, each carrying some information about the input.
// So in grayscale images for example, we have 1 channel that just tells the brightness of each pixel,
// while in RGB images, we have three channels (Red, Green, And Blue Layers) and after convolution, each filter produces a new channel.
// Each channel is basically a feature map.
// If that didnt make sense, this might:
// A grayscale image has only 1 intensity per pixel, no color layers. Since there is only one piece of information per pixel, C_in = 1
// A color image has 3 layers (RGB) => C_in = 3. 
// So C_in is determined by your data
// IF that still didnt make sense, go to youtube or sumn

// input: [N, C_in, H, W], where N => Batch Size (Number of Images), C_in => input channels (Eg: 1 for MNIST), H => Height of the input, W => Width of the input
// weight: [C_out, C_in, K, K], where C_out => Number of filters/output channels, K=> Kernel/Filter Size (Ex: 3X3)
// output: [N, C_out, H_out, W_out], where H_out => Height After Convolution, W => Weight After Convolution
// stride is how much the filter moves each step. Ex: Stride = 1 means slide one pixel at a time
// padding is basically optional extra pixels around the image. This keeps output size the same as input.
// padding is usually 0, or atleast that is the lazy way of doing it, as ali mentioned
__global__ void conv2d_simple_kernel(
    const float* input, const float* weight, const float* bias, float* output,
    int N, int C_in, int H, int W, int C_out, int K, int pad, int stride, int H_out, int W_out
){
    int globalID = threadIdx.x + (blockIdx.x * blockDim.x);
    int total = N * C_out * H_out * W_out;

    if (globalID >= total){
        return;
    }

    // Compute 4D indices from flat index
    int tmp = globalID;
    
    int x = tmp % W_out; 
    tmp /= W_out;
    
    int y = tmp % H_out; 
    tmp /= H_out;
    
    int m = tmp % C_out; 
    tmp /= C_out;

    int n = tmp; // batch index

    float acc = 0.0f;
    for(int c = 0; c < C_in; c++){
        for(int ky = 0; ky < K; ky++){
            for(int kx = 0; kx < K; kx++){
                int in_y = (y * stride) + ky - pad;
                int in_x = (x * stride) + kx - pad;
                if(in_y >= 0 && in_y < H && in_x >= 0 && in_x < W){
                    int in_idx = (n * C_in * H * W) + (c * H * W) + (in_y * W) + in_x;
                    int w_idx = (m * C_in * K * K) + (c * K * K) + (ky * K) + kx;
                    acc += input[in_idx] * weight[w_idx];
                }
            }
        }
    }

    if(bias){
        acc += bias[m];
    }

    int out_idx = (n * C_out * H_out * W_out) + (m * H_out * W_out) + (y * W_out) + x;
    output[out_idx] = acc;
}

// ReLU activation
__global__ void relu_kernel(const float* in, float* out, int N){
    int threadID = threadIdx.x + (blockIdx.x * blockDim.x);
    if(threadID >= N){
        return;
    }
    float v = in[threadID];
    out[threadID] = v>0 ? v:0; // max(0,v)
}

// MaxPool 2x2
__global__ void maxpool2x2_kernel(const float* in, float* out, int N, int C, int H, int W, int H_out, int W_out){
    int globalID = threadIdx.x + (blockIdx.x + blockDim.x);
    int total = N * C * H_out * W_out;
    
    if (globalID >= total){
        return;
    }

    int tmp = globalID;

    int x = tmp%W_out; 
    tmp /= W_out;

    int y = tmp % H_out; 
    tmp /= H_out;

    int c = tmp%C; 
    tmp /= C;

    int n = tmp;

    int in_y0 = y * 2;
    int in_x0 = x * 2;
    float mval =- 1e30;

    for(int dy = 0; dy < 2; dy++){

        for(int dx = 0; dx < 2; dx++){

            int iy = in_y0 + dy;
            int ix = in_x0 + dx;
            int idx2 = (n * C * H * W) + (c * H * W) + (iy * W) + ix;
            float val = in[idx2];
            if(val > mval){
                mval = val;
            } 
        }
    }
    int out_idx = (n * C * H_out * W_out) + (c * H_out * W_out) + (y * W_out) + x;
    out[out_idx] = mval;
}

// Fully connected forward
__global__ void fc_forward_kernel(const float* in, const float* W, const float* b, float* out,int Nn,int D,int O){
    int globalID = threadIdx.x + (blockIdx.x * blockDim.x);
    int total = Nn * O;
    if(globalID >= total){
        return;
    }

    int o = globalID % O;
    int n = globalID / O;

    float acc = 0;
    for(int i = 0; i < D; i++){
        acc += (in[n * D + i]) * (W[(o * D) + i]);
    }
    if(b) {
        acc += b[o];
    }
    out[(n * O) + o] = acc;
}
