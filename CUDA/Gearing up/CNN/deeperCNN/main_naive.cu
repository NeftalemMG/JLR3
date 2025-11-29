%%writefile main_naive.cu
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

// ---------- CUDA error check ----------
static void checkCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

// ---------- Load MNIST helpers ----------
uint32_t read_u32_be(FILE *f) {
    unsigned char b[4];
    fread(b,1,4,f);
    return (b[0]<<24)|(b[1]<<16)|(b[2]<<8)|b[3];
}

unsigned char* load_idx_images(const char *path, int *outN, int *outH, int *outW) {
    FILE *f = fopen(path,"rb");
    if (!f) { fprintf(stderr,"Failed to open %s\n", path); return NULL; }
    read_u32_be(f); // magic
    uint32_t N = read_u32_be(f);
    uint32_t H = read_u32_be(f);
    uint32_t W = read_u32_be(f);
    unsigned char *buf = (unsigned char*)malloc(N*H*W);
    fread(buf,1,N*H*W,f);
    fclose(f);
    *outN=N; *outH=H; *outW=W;
    return buf;
}

unsigned char* load_idx_labels(const char *path, int *outN) {
    FILE *f = fopen(path,"rb");
    if (!f) { fprintf(stderr,"Failed to open %s\n", path); return NULL; }
    read_u32_be(f); // magic
    uint32_t N = read_u32_be(f);
    unsigned char *buf = (unsigned char*)malloc(N);
    fread(buf,1,N,f);
    fclose(f);
    *outN=N;
    return buf;
}

// Convert uint8 images to float [-1,1] normalized
void convert_images_u8_to_f32(unsigned char *u8, float *out, int N, int H, int W) {
    for(int i=0;i<N*H*W;i++)
        out[i] = ((float)u8[i]/255.0f - 0.1307f)/0.3081f;
}

// ---------- Simple RNG ----------
float frand() { return ((float)rand()/RAND_MAX)*2-1; }

// ---------- CUDA Kernels ----------

// Naive convolution
__global__ void conv2d_naive_kernel(
    const float* input, const float* weight, const float* bias, float* output,
    int N, int C_in, int H, int W, int C_out, int K, int pad, int stride, int H_out, int W_out
){
    int gid = blockIdx.x*blockDim.x+threadIdx.x;
    int total = N*C_out*H_out*W_out;
    if(gid>=total) return;
    
    int tmp=gid;
    int x=tmp%W_out; tmp/=W_out;
    int y=tmp%H_out; tmp/=H_out;
    int m=tmp%C_out; tmp/=C_out;
    int n=tmp;

    float acc=0.0f;
    for(int c=0;c<C_in;c++){
        for(int ky=0;ky<K;ky++){
            for(int kx=0;kx<K;kx++){
                int in_y=y*stride+ky-pad;
                int in_x=x*stride+kx-pad;
                if(in_y>=0 && in_y<H && in_x>=0 && in_x<W){
                    int in_idx = n*C_in*H*W + c*H*W + in_y*W + in_x;
                    int w_idx = m*C_in*K*K + c*K*K + ky*K + kx;
                    acc += input[in_idx]*weight[w_idx];
                }
            }
        }
    }
    if(bias) acc += bias[m];
    int out_idx = n*C_out*H_out*W_out + m*H_out*W_out + y*W_out + x;
    output[out_idx]=acc;
}

// ReLU
__global__ void relu_kernel(const float* in, float* out, int N) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=N) return;
    float v = in[idx];
    out[idx] = v>0?v:0;
}

// MaxPool 2x2
__global__ void maxpool2x2_kernel(const float* in, float* out, int N, int C, int H, int W, int H_out, int W_out){
    int gid = blockIdx.x*blockDim.x+threadIdx.x;
    int total = N*C*H_out*W_out;
    if(gid>=total) return;
    int tmp=gid;
    int x=tmp%W_out; tmp/=W_out;
    int y=tmp%H_out; tmp/=H_out;
    int c=tmp%C; tmp/=C;
    int n=tmp;

    int in_y0=y*2, in_x0=x*2;
    float mval=-1e30;
    for(int dy=0;dy<2;dy++)
        for(int dx=0;dx<2;dx++){
            int iy=in_y0+dy, ix=in_x0+dx;
            int idx2=n*C*H*W + c*H*W + iy*W + ix;
            float val=in[idx2];
            if(val>mval) mval=val;
        }
    int out_idx=n*C*H_out*W_out + c*H_out*W_out + y*W_out + x;
    out[out_idx]=mval;
}

// Fully connected
__global__ void fc_forward_kernel(const float* in, const float* W, const float* b, float* out,int Nn,int D,int O){
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    int total=Nn*O;
    if(gid>=total) return;
    int o=gid%O, n=gid/O;
    float acc=0;
    for(int i=0;i<D;i++) acc+=in[n*D+i]*W[o*D+i];
    if(b) acc+=b[o];
    out[n*O+o]=acc;
}

// ---------- Host softmax & cross-entropy ----------
void softmax_and_loss_host(const float* logits, const unsigned char* labels,int N,int O,float* out_probs,float* out_loss){
    float total_loss=0;
    for(int n=0;n<N;n++){
        const float* row = logits + n*O;
        float m=row[0];
        for(int j=1;j<O;j++) if(row[j]>m)m=row[j];
        double sum=0;
        for(int j=0;j<O;j++) sum+=exp((double)row[j]-m);
        for(int j=0;j<O;j++) out_probs[n*O+j]=(float)(exp((double)row[j]-m)/sum);
        int t=labels[n];
        total_loss += -logf(out_probs[n*O+t]+1e-12f);
    }
    *out_loss = total_loss/(float)N;
}

// ---------- Host utility ----------
void init_weights(float* W,int n){for(int i=0;i<n;i++) W[i]=0.01f*frand();}
void zero_fill(float* p,int n){for(int i=0;i<n;i++) p[i]=0;}

// ---------- Main ----------
int main(){
    srand(1234);

    // 1) Load MNIST dataset
    int trainN,H,W;
    unsigned char* train_u8 = load_idx_images("/kaggle/input/mnist-dataset/train-images.idx3-ubyte",&trainN,&H,&W);
    int trainLabelsN;
    unsigned char* train_labels=load_idx_labels("/kaggle/input/mnist-dataset/train-labels.idx1-ubyte",&trainLabelsN);
    if(!train_u8){printf("Failed to load images\n"); return 1;}
    printf("Loaded train images: N=%d H=%d W=%d\n",trainN,H,W);

    // Use small subset for demo
    int N=64;
    float *h_input = (float*)malloc(sizeof(float)*N*H*W);
    convert_images_u8_to_f32(train_u8,h_input,N,H,W);

    unsigned char* h_labels = (unsigned char*)malloc(N);
    for(int i=0;i<N;i++) h_labels[i]=train_labels[i];

    free(train_u8); free(train_labels);

    // 2) Model parameters
    int Cin=1, conv_out=16, K=3, pad=1, stride=1;
    int H_out=(H+2*pad-K)/stride+1, W_out=(W+2*pad-K)/stride+1;
    int H_pool=H_out/2, W_pool=W_out/2;
    int fc_in_dim=conv_out*H_pool*W_pool, num_classes=10;

    // Allocate & init weights
    float *h_conv_w=(float*)malloc(sizeof(float)*conv_out*Cin*K*K);
    float *h_conv_b=(float*)malloc(sizeof(float)*conv_out);
    init_weights(h_conv_w,conv_out*Cin*K*K); zero_fill(h_conv_b,conv_out);

    float *h_fc_w=(float*)malloc(sizeof(float)*num_classes*fc_in_dim);
    float *h_fc_b=(float*)malloc(sizeof(float)*num_classes);
    init_weights(h_fc_w,num_classes*fc_in_dim); zero_fill(h_fc_b,num_classes);

    // 3) Allocate device memory
    float *d_input,*d_conv_w,*d_conv_b,*d_conv_out,*d_pool_out,*d_fc_in,*d_fc_w,*d_fc_b,*d_logits;
    checkCuda(cudaMalloc(&d_input,sizeof(float)*N*H*W),"d_input");
    checkCuda(cudaMemcpy(d_input,h_input,sizeof(float)*N*H*W,cudaMemcpyHostToDevice),"H2D input");
    checkCuda(cudaMalloc(&d_conv_w,sizeof(float)*conv_out*Cin*K*K),"d_conv_w");
    checkCuda(cudaMemcpy(d_conv_w,h_conv_w,sizeof(float)*conv_out*Cin*K*K,cudaMemcpyHostToDevice),"H2D conv_w");
    checkCuda(cudaMalloc(&d_conv_b,sizeof(float)*conv_out),"d_conv_b");
    checkCuda(cudaMemcpy(d_conv_b,h_conv_b,sizeof(float)*conv_out,cudaMemcpyHostToDevice),"H2D conv_b");

    checkCuda(cudaMalloc(&d_conv_out,sizeof(float)*N*conv_out*H_out*W_out),"d_conv_out");
    checkCuda(cudaMalloc(&d_pool_out,sizeof(float)*N*conv_out*H_pool*W_pool),"d_pool_out");

    checkCuda(cudaMalloc(&d_fc_in,sizeof(float)*N*fc_in_dim),"d_fc_in");
    checkCuda(cudaMalloc(&d_fc_w,sizeof(float)*num_classes*fc_in_dim),"d_fc_w");
    checkCuda(cudaMemcpy(d_fc_w,h_fc_w,sizeof(float)*num_classes*fc_in_dim,cudaMemcpyHostToDevice),"H2D fc_w");
    checkCuda(cudaMalloc(&d_fc_b,sizeof(float)*num_classes),"d_fc_b");
    checkCuda(cudaMemcpy(d_fc_b,h_fc_b,sizeof(float)*num_classes,cudaMemcpyHostToDevice),"H2D fc_b");

    checkCuda(cudaMalloc(&d_logits,sizeof(float)*N*num_classes),"d_logits");

    // 4) Forward + training demo (tiny 2 epochs)
    int threads=256;
    for(int epoch=0;epoch<2;epoch++){
        // Conv
        int total_conv=N*conv_out*H_out*W_out;
        int blocks=(total_conv+threads-1)/threads;
        conv2d_naive_kernel<<<blocks,threads>>>(d_input,d_conv_w,d_conv_b,d_conv_out,N,Cin,H,W,conv_out,K,pad,stride,H_out,W_out);
        cudaDeviceSynchronize();

        // ReLU
        relu_kernel<<<blocks,threads>>>(d_conv_out,d_conv_out,total_conv);
        cudaDeviceSynchronize();

        // MaxPool
        int total_pool=N*conv_out*H_pool*W_pool;
        blocks=(total_pool+threads-1)/threads;
        maxpool2x2_kernel<<<blocks,threads>>>(d_conv_out,d_pool_out,N,conv_out,H_out,W_out,H_pool,W_pool);
        cudaDeviceSynchronize();

        // Flatten for FC (device-to-device copy)
        for(int n=0;n<N;n++){
            checkCuda(cudaMemcpy(d_fc_in+n*fc_in_dim,d_pool_out+n*conv_out*H_pool*W_pool,sizeof(float)*conv_out*H_pool*W_pool,cudaMemcpyDeviceToDevice),"flatten");
        }

        // FC
        blocks=(N*num_classes+threads-1)/threads;
        fc_forward_kernel<<<blocks,threads>>>(d_fc_in,d_fc_w,d_fc_b,d_logits,N,fc_in_dim,num_classes);
        cudaDeviceSynchronize();

        // Copy logits to host & compute softmax + loss
        float *h_logits=(float*)malloc(sizeof(float)*N*num_classes);
        float *h_probs=(float*)malloc(sizeof(float)*N*num_classes);
        float loss=0;
        checkCuda(cudaMemcpy(h_logits,d_logits,sizeof(float)*N*num_classes,cudaMemcpyDeviceToHost),"D2H logits");
        softmax_and_loss_host(h_logits,h_labels,N,num_classes,h_probs,&loss);

        printf("Epoch %d: loss=%f\n",epoch,loss);

        // Compute accuracy
        int correct=0;
        for(int i=0;i<N;i++){
            int pred=0; float maxp=h_probs[i*num_classes];
            for(int j=1;j<num_classes;j++) if(h_probs[i*num_classes+j]>maxp){maxp=h_probs[i*num_classes+j]; pred=j;}
            if(pred==h_labels[i]) correct++;
            printf("Sample %d: true=%d pred=%d prob=%f\n",i,h_labels[i],pred,maxp);
        }
        printf("Epoch %d: accuracy=%f\n",epoch,(float)correct/N);

        free(h_logits); free(h_probs);
    }

    // Cleanup
    cudaFree(d_input); cudaFree(d_conv_w); cudaFree(d_conv_b); cudaFree(d_conv_out);
    cudaFree(d_pool_out); cudaFree(d_fc_in); cudaFree(d_fc_w); cudaFree(d_fc_b); cudaFree(d_logits);
    free(h_input); free(h_labels); free(h_conv_w); free(h_conv_b); free(h_fc_w); free(h_fc_b);

    return 0;
}
