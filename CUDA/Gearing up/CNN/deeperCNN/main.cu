%%writefile main.cu
// main.cu - single-file CUDA C example (naive CNN forward + simple FC update demo) for MNIST
// Builds two model flavors:
//   Option A: Single conv (Conv->ReLU->Pool->FC->Softmax)  [fast to test]
//   Option B: Two convs (Conv->ReLU->Pool->Conv->ReLU->Pool->FC->Softmax) [slower]
// Usage: ./main --mode quick    (quick run with tiny subset)
//        ./main --mode eval     (run full test set inference; may be slow on Kaggle)

// NOTE: This file focuses on forward inference kernels + loss. It demonstrates a small SGD
// update on the final fully-connected layer on the host to show training mechanics.
// Full backprop for conv layers is large and left as the next step (I explain how to do it).

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvToolsExt.h>    // NVTX markers for profiling tools (optional)

// ---------- utility helpers ----------
static void checkCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

uint32_t read_u32_be(FILE *f) {
    unsigned char b[4];
    if (fread(b,1,4,f) != 4) { fprintf(stderr,"read error\n"); exit(1); }
    return (b[0]<<24) | (b[1]<<16) | (b[2]<<8) | b[3];
}

// load MNIST IDX images (uint8) -> returns pointer to contiguous uint8 buffer (N*H*W)
unsigned char* load_idx_images(const char *path, int *outN, int *outH, int *outW) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr,"Failed to open %s\n", path); return NULL; }
    uint32_t magic = read_u32_be(f);
    uint32_t N = read_u32_be(f);
    uint32_t H = read_u32_be(f);
    uint32_t W = read_u32_be(f);
    // magic should be 2051 for images
    unsigned char *buf = (unsigned char*)malloc(N * H * W);
    size_t got = fread(buf, 1, (size_t)N*H*W, f);
    if (got != (size_t)N*H*W) {
        fprintf(stderr,"IDX read mismatch: got %zu expected %zu\n", got, (size_t)N*H*W);
        free(buf); fclose(f); return NULL;
    }
    fclose(f);
    *outN = (int)N; *outH = (int)H; *outW = (int)W;
    return buf;
}

// load MNIST labels (uint8)
unsigned char* load_idx_labels(const char *path, int *outN) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr,"Failed to open %s\n", path); return NULL; }
    uint32_t magic = read_u32_be(f);
    uint32_t N = read_u32_be(f);
    unsigned char *buf = (unsigned char*)malloc(N);
    size_t got = fread(buf,1,(size_t)N,f);
    if (got != (size_t)N) { fprintf(stderr,"label read mismatch\n"); free(buf); fclose(f); return NULL; }
    fclose(f);
    *outN = (int)N;
    return buf;
}

// convert uint8 images to float normalized [-1,1]
void convert_images_u8_to_f32(unsigned char *u8, float *out, int N, int H, int W) {
    int size = N * H * W;
    for (int i = 0; i < size; ++i) {
        out[i] = ((float)u8[i] / 255.0f - 0.1307f) / 0.3081f; // same normalization as torchvision MNIST (mean,std)
    }
}

// ---------- simple RNG ----------
float frand() { return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f; }

// ---------- naive conv kernel (one thread per output element) ----------
__global__ void conv2d_naive_kernel(
    const float* __restrict__ input,   // N * C_in * H * W
    const float* __restrict__ weight,  // C_out * C_in * K * K
    const float* __restrict__ bias,    // C_out or NULL
    float* __restrict__ output,        // N * C_out * H_out * W_out
    int N, int C_in, int H, int W,
    int C_out, int K, int pad, int stride,
    int H_out, int W_out)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H_out * W_out;
    if (gid >= total) return;
    int tmp = gid;
    int x = tmp % W_out; tmp /= W_out;
    int y = tmp % H_out; tmp /= H_out;
    int m = tmp % C_out; tmp /= C_out;
    int n = tmp; // remaining

    float acc = 0.0f;
    for (int c = 0; c < C_in; ++c) {
        for (int ky = 0; ky < K; ++ky) {
            for (int kx = 0; kx < K; ++kx) {
                int in_y = y * stride + ky - pad;
                int in_x = x * stride + kx - pad;
                if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                    int in_idx  = n*C_in*H*W + c*H*W + in_y*W + in_x;
                    int w_idx   = m*C_in*K*K + c*K*K + ky*K + kx;
                    acc += input[in_idx] * weight[w_idx];
                }
            }
        }
    }
    if (bias) acc += bias[m];
    int out_idx = n*C_out*H_out*W_out + m*H_out*W_out + y*W_out + x;
    output[out_idx] = acc;
}

// ---------- ReLU kernel ----------
__global__ void relu_kernel(const float *in, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float v = in[idx];
    out[idx] = v > 0.0f ? v : 0.0f;
}

// ---------- MaxPool kernel (2x2, stride=2) - naive per-output thread (each output element maps to 2x2 region) ----------
__global__ void maxpool2x2_kernel(const float *in, float *out, int N, int C, int H, int W, int H_out, int W_out) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (gid >= total) return;
    int tmp = gid;
    int x = tmp % W_out; tmp /= W_out;
    int y = tmp % H_out; tmp /= H_out;
    int c = tmp % C; tmp /= C;
    int n = tmp;
    int in_y0 = y * 2;
    int in_x0 = x * 2;
    float m = -1e30f;
    for (int dy=0; dy<2; ++dy) for (int dx=0; dx<2; ++dx) {
        int iy = in_y0 + dy;
        int ix = in_x0 + dx;
        int idx = n*C*H*W + c*H*W + iy*W + ix;
        float val = in[idx];
        if (val > m) m = val;
    }
    int out_idx = n*C*H_out*W_out + c*H_out*W_out + y*W_out + x;
    out[out_idx] = m;
}

// ---------- Fully-connected forward kernel (matrix multiply naive) ----------
__global__ void fc_forward_kernel(
    const float *in,    // N x D  (flattened: N*D)
    const float *W,     // O x D  (flattened row-major: O*D)
    const float *b,     // O
    float *out,         // N x O
    int Nn, int D, int O)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nn * O;
    if (gid >= total) return;
    int o = gid % O;
    int n = gid / O;
    float acc = 0.0f;
    const float *Wrow = W + o * D;
    const float *inrow = in + n * D;
    for (int i=0;i<D;++i) acc += inrow[i] * Wrow[i];
    if (b) acc += b[o];
    out[n*O + o] = acc;
}

// ---------- Softmax + cross-entropy (compute softmax per sample and loss on host for stability) ----------
// We'll compute logits on device, copy to host, compute softmax+loss per-sample on host (numerically stable) for simplicity.
// (You could implement numerically-stable softmax on device with reductions; that's next-step.)
//
// small helper: host-side softmax & loss
void softmax_and_loss_host(const float *logits, const unsigned char *labels, int N, int O, float *out_probs, float *out_loss) {
    float total_loss = 0.0f;
    for (int n=0;n<N;++n) {
        const float *row = logits + n*O;
        // find max
        float m = row[0];
        for (int j=1;j<O;++j) if (row[j] > m) m = row[j];
        double sum = 0.0;
        for (int j=0;j<O;++j) sum += exp((double)row[j] - (double)m);
        for (int j=0;j<O;++j) out_probs[n*O + j] = (float)(exp((double)row[j] - (double)m) / sum);
        int t = labels[n];
        float ls = -logf(out_probs[n*O + t] + 1e-12f);
        total_loss += ls;
    }
    *out_loss = total_loss / (float)N;
}

// ---------- Host model utilities (weights init, etc.) ----------
void init_weights(float *W, int n) { for (int i=0;i<n;++i) W[i] = 0.01f * frand(); }
void zero_fill(float *p, int n) { for (int i=0;i<n;++i) p[i] = 0.0f; }

// ---------- Main: loads data, sets up model, runs forward/mini training ----------------
int main(int argc, char** argv) {
    srand(1234);
    // parse mode
    const char *mode = "quick";
    if (argc >= 2) {
        if (strcmp(argv[1],"--mode")==0 && argc>=3) mode = argv[2];
    }

    // 1) Load dataset (copy dataset to working dir first; see notebook step earlier)
    int trainN=0, H=0, W=0;
    unsigned char *train_u8 = load_idx_images("/kaggle/input/mnist-dataset/train-images.idx3-ubyte", &trainN, &H, &W);
    int trainLabelsN=0;
    unsigned char *train_labels = load_idx_labels("/kaggle/input/mnist-dataset/train-labels.idx1-ubyte", &trainLabelsN);
    if (!train_u8) { fprintf(stderr,"Failed to load images\n"); return 1; }
    printf("Loaded train images: N=%d H=%d W=%d\n", trainN, H, W);

    // quickmode: use small subset to verify
    int useN = (strcmp(mode,"quick")==0) ? 64 : trainN; // quick uses 64 images
    if (useN > trainN) useN = trainN;

    // allocate float input and normalize
    int Cin = 1;
    int N = useN;
    int img_size = N * Cin * H * W;
    float *h_input = (float*)malloc(sizeof(float) * img_size);
    convert_images_u8_to_f32(train_u8, h_input, N, H, W);

    // labels subset
    unsigned char *h_labels = (unsigned char*)malloc(N);
    for (int i=0;i<N;++i) h_labels[i] = train_labels[i];

    // free raw u8 if needed
    free(train_u8); // we copied needed subset into h_input

    // ---------- Model configuration ----------
    // Option A: single conv -> relu -> pool -> fc
    // Option B: two convs -> pool -> fc
    int option = 2; // 1 => option A (single conv), 2 => option B (two convs)
    // Conv params
    int K = 3;
    int pad = 1;
    int stride = 1;
    int conv1_out = 16; // C_out1
    int conv2_out = 32; // C_out2 (used only for option B)
    // After first conv + pool (2x2) -> H/2 x W/2
    int H1 = H; int W1 = W;
    int H1_out = (H1 + 2*pad - K)/stride + 1;
    int W1_out = (W1 + 2*pad - K)/stride + 1;
    int H1_p = H1_out / 2; int W1_p = W1_out / 2;

    int H2_out=0, W2_out=0, H2_p=0, W2_p=0;
    if (option==2) {
        H2_out = (H1_p + 2*pad - K)/stride + 1;
        W2_out = (W1_p + 2*pad - K)/stride + 1;
        H2_p = H2_out / 2; W2_p = W2_out / 2;
    }

    // FC input dimension and output classes
    int fc_in_dim = (option==1) ? (conv1_out * H1_p * W1_p) : (conv2_out * H2_p * W2_p);
    int num_classes = 10;

    printf("Model config: option=%d conv1_out=%d conv2_out=%d fc_in=%d\n", option, conv1_out, conv2_out, fc_in_dim);

    // ---------- Allocate and initialize weights on host ----------
    // conv1 weights: conv1_out * Cin * K * K
    int conv1_w_size = conv1_out * Cin * K * K;
    float *h_conv1_w = (float*)malloc(sizeof(float) * conv1_w_size);
    float *h_conv1_b = (float*)malloc(sizeof(float) * conv1_out);
    init_weights(h_conv1_w, conv1_w_size); zero_fill(h_conv1_b, conv1_out);

    // conv2 weights (if option==2)
    int conv2_w_size = conv2_out * conv1_out * K * K;
    float *h_conv2_w = NULL; float *h_conv2_b = NULL;
    if (option==2) {
        h_conv2_w = (float*)malloc(sizeof(float) * conv2_w_size);
        h_conv2_b = (float*)malloc(sizeof(float) * conv2_out);
        init_weights(h_conv2_w, conv2_w_size); zero_fill(h_conv2_b, conv2_out);
    }

    // FC weights: num_classes x fc_in_dim
    int fc_w_size = num_classes * fc_in_dim;
    float *h_fc_w = (float*)malloc(sizeof(float) * fc_w_size);
    float *h_fc_b = (float*)malloc(sizeof(float) * num_classes);
    init_weights(h_fc_w, fc_w_size); zero_fill(h_fc_b, num_classes);

    // ---------- Move data & weights to device ----------
    float *d_input = NULL;
    checkCuda(cudaMalloc((void**)&d_input, sizeof(float) * img_size), "cudaMalloc d_input");
    checkCuda(cudaMemcpy(d_input, h_input, sizeof(float)*img_size, cudaMemcpyHostToDevice), "memcpy H2D input");

    float *d_conv1_w = NULL, *d_conv1_b = NULL;
    checkCuda(cudaMalloc((void**)&d_conv1_w, sizeof(float) * conv1_w_size), "cudaMalloc conv1_w");
    checkCuda(cudaMemcpy(d_conv1_w, h_conv1_w, sizeof(float)*conv1_w_size, cudaMemcpyHostToDevice), "memcpy conv1_w");
    checkCuda(cudaMalloc((void**)&d_conv1_b, sizeof(float) * conv1_out), "cudaMalloc conv1_b");
    checkCuda(cudaMemcpy(d_conv1_b, h_conv1_b, sizeof(float)*conv1_out, cudaMemcpyHostToDevice), "memcpy conv1_b");

    float *d_conv2_w = NULL, *d_conv2_b = NULL;
    if (option==2) {
        checkCuda(cudaMalloc((void**)&d_conv2_w, sizeof(float) * conv2_w_size), "cudaMalloc conv2_w");
        checkCuda(cudaMemcpy(d_conv2_w, h_conv2_w, sizeof(float)*conv2_w_size, cudaMemcpyHostToDevice), "memcpy conv2_w");
        checkCuda(cudaMalloc((void**)&d_conv2_b, sizeof(float) * conv2_out), "cudaMalloc conv2_b");
        checkCuda(cudaMemcpy(d_conv2_b, h_conv2_b, sizeof(float)*conv2_out, cudaMemcpyHostToDevice), "memcpy conv2_b");
    }

    float *d_fc_w = NULL, *d_fc_b = NULL;
    checkCuda(cudaMalloc((void**)&d_fc_w, sizeof(float) * fc_w_size), "cudaMalloc fc_w");
    checkCuda(cudaMemcpy(d_fc_w, h_fc_w, sizeof(float)*fc_w_size, cudaMemcpyHostToDevice), "memcpy fc_w");
    checkCuda(cudaMalloc((void**)&d_fc_b, sizeof(float) * num_classes), "cudaMalloc fc_b");
    checkCuda(cudaMemcpy(d_fc_b, h_fc_b, sizeof(float)*num_classes, cudaMemcpyHostToDevice), "memcpy fc_b");

    // ---------- Buffers for intermediate activations ----------
    // conv1 output: N * conv1_out * H1_out * W1_out
    int conv1_out_elems = N * conv1_out * H1_out * W1_out;
    float *d_conv1_out = NULL;
    checkCuda(cudaMalloc((void**)&d_conv1_out, sizeof(float) * conv1_out_elems), "cudaMalloc conv1_out");

    // after pool1: N * conv1_out * H1_p * W1_p
    int pool1_elems = N * conv1_out * H1_p * W1_p;
    float *d_pool1_out = NULL;
    checkCuda(cudaMalloc((void**)&d_pool1_out, sizeof(float) * pool1_elems), "cudaMalloc pool1_out");

    // conv2 & pool2 buffers if option==2
    float *d_conv2_out = NULL, *d_pool2_out = NULL;
    if (option==2) {
        checkCuda(cudaMalloc((void**)&d_conv2_out, sizeof(float) * N * conv2_out * H2_out * W2_out), "cudaMalloc conv2_out");
        checkCuda(cudaMalloc((void**)&d_pool2_out, sizeof(float) * N * conv2_out * H2_p * W2_p), "cudaMalloc pool2_out");
    }

    // flatten buffer for FC input: N x fc_in_dim
    float *d_fc_in = NULL;
    checkCuda(cudaMalloc((void**)&d_fc_in, sizeof(float) * N * fc_in_dim), "cudaMalloc fc_in");

    // FC output logits: N x num_classes
    float *d_logits = NULL;
    checkCuda(cudaMalloc((void**)&d_logits, sizeof(float) * N * num_classes), "cudaMalloc logits");

    // temporary host buffers for loss/softmax (we compute softmax on host for numeric stability)
    float *h_logits = (float*)malloc(sizeof(float) * N * num_classes);
    float *h_probs  = (float*)malloc(sizeof(float) * N * num_classes);
    float total_loss = 0.0f;

    // ---------- RUN forward: conv1 -> relu -> pool1 -> (conv2->relu->pool2) -> flatten -> fc -> logits ----------
    // We'll run a single forward pass and compute loss on host, then demonstrate a tiny SGD step on FC weights on host.

    // conv1 launch
    nvtxRangePushA("conv1_forward");
    {
        int threads = 256;
        int total_out = N * conv1_out * H1_out * W1_out;
        int blocks = (total_out + threads - 1) / threads;
        conv2d_naive_kernel<<<blocks, threads>>>(d_input, d_conv1_w, d_conv1_b, d_conv1_out,
            N, Cin, H, W, conv1_out, K, pad, stride, H1_out, W1_out);
        checkCuda(cudaGetLastError(), "launch conv1");
    }
    nvtxRangePop();

    // relu1
    nvtxRangePushA("relu1");
    {
        int threads = 256;
        int total = conv1_out_elems;
        int blocks = (total + threads - 1) / threads;
        relu_kernel<<<blocks, threads>>>(d_conv1_out, d_conv1_out, total); // in-place
        checkCuda(cudaGetLastError(), "launch relu1");
    }
    nvtxRangePop();

    // pool1 (2x2)
    nvtxRangePushA("pool1");
    {
        int threads = 256;
        int total = pool1_elems;
        int blocks = (total + threads - 1) / threads;
        maxpool2x2_kernel<<<blocks, threads>>>(d_conv1_out, d_pool1_out, N, conv1_out, H1_out, W1_out, H1_p, W1_p);
        checkCuda(cudaGetLastError(), "launch pool1");
    }
    nvtxRangePop();

    // If option==2, conv2 -> relu -> pool2
    if (option==2) {
        nvtxRangePushA("conv2_forward");
        {
            int threads = 256;
            int total_out = N * conv2_out * H2_out * W2_out;
            int blocks = (total_out + threads - 1) / threads;
            conv2d_naive_kernel<<<blocks, threads>>>(
                d_pool1_out, d_conv2_w, d_conv2_b, d_conv2_out,
                N, conv1_out, H1_p, W1_p, conv2_out, K, pad, stride, H2_out, W2_out);
            checkCuda(cudaGetLastError(), "launch conv2");
        }
        nvtxRangePop();

        nvtxRangePushA("relu2");
        {
            int threads = 256;
            int total = N * conv2_out * H2_out * W2_out;
            int blocks = (total + threads - 1) / threads;
            relu_kernel<<<blocks, threads>>>(d_conv2_out, d_conv2_out, total);
            checkCuda(cudaGetLastError(), "launch relu2");
        }
        nvtxRangePop();

        nvtxRangePushA("pool2");
        {
            int threads = 256;
            int total = N * conv2_out * H2_p * W2_p;
            int blocks = (total + threads - 1) / threads;
            maxpool2x2_kernel<<<blocks, threads>>>(d_conv2_out, d_pool2_out, N, conv2_out, H2_out, W2_out, H2_p, W2_p);
            checkCuda(cudaGetLastError(), "launch pool2");
        }
        nvtxRangePop();

        // flatten pool2 -> fc_in (N x fc_in_dim)
        // naive copy kernel on host: we'll copy device memory region by region via cudaMemcpy per-image (simple)
        for (int n=0; n<N; ++n) {
            size_t src_off = (size_t)n * conv2_out * H2_p * W2_p * sizeof(float);
            size_t dst_off = (size_t)n * conv2_out * H2_p * W2_p * sizeof(float);
            checkCuda(cudaMemcpy((char*)d_fc_in + dst_off, (char*)d_pool2_out + src_off, conv2_out * H2_p * W2_p * sizeof(float), cudaMemcpyDeviceToDevice), "memcpy flatten");
        }
    } else {
        // option 1 flatten pool1 -> fc_in
        for (int n=0; n<N; ++n) {
            size_t src_off = (size_t)n * conv1_out * H1_p * W1_p * sizeof(float);
            size_t dst_off = (size_t)n * conv1_out * H1_p * W1_p * sizeof(float);
            checkCuda(cudaMemcpy((char*)d_fc_in + dst_off, (char*)d_pool1_out + src_off, conv1_out * H1_p * W1_p * sizeof(float), cudaMemcpyDeviceToDevice), "memcpy flatten1");
        }
    }

    // FC forward
    nvtxRangePushA("fc_forward");
    {
        int threads = 256;
        int total = N * num_classes;
        int blocks = (total + threads - 1) / threads;
        fc_forward_kernel<<<blocks, threads>>>(d_fc_in, d_fc_w, d_fc_b, d_logits, N, fc_in_dim, num_classes);
        checkCuda(cudaGetLastError(), "launch fc");
    }
    nvtxRangePop();

    // Copy logits to host and compute softmax + loss on host (numerical stability)
    checkCuda(cudaMemcpy(h_logits, d_logits, sizeof(float) * N * num_classes, cudaMemcpyDeviceToHost), "memcpy logits D2H");
    softmax_and_loss_host(h_logits, h_labels, N, num_classes, h_probs, &total_loss);
    printf("Forward pass done. N=%d loss(avg)=%f\n", N, total_loss);

    // Demonstration: a tiny SGD update on the final FC weights on HOST (VERY simple, NOT full backprop)
    // We'll compute gradient for FC only (dL/dz = probs - one_hot), then dW = (1/N) * dLdz^T * fc_input
    // Copy fc_in to host:
    float *h_fc_in = (float*)malloc(sizeof(float) * N * fc_in_dim);
    checkCuda(cudaMemcpy(h_fc_in, d_fc_in, sizeof(float) * N * fc_in_dim, cudaMemcpyDeviceToHost), "memcpy fc_in D2H");

    // compute dL/dz = probs - y_onehot
    float *dLdz = (float*)malloc(sizeof(float) * N * num_classes);
    for (int n=0;n<N;++n) {
        for (int o=0;o<num_classes;++o) {
            float p = h_probs[n*num_classes + o];
            int t = (int)h_labels[n];
            dLdz[n*num_classes + o] = p - (o==t ? 1.0f : 0.0f);
        }
    }
    // compute gradient w.r.t fc weights: dW[o,d] = (1/N) * sum_n dLdz[n,o] * fc_in[n,d]
    float lr = 0.01f;
    for (int o=0;o<num_classes;++o) {
        for (int d=0; d<fc_in_dim; ++d) {
            double acc = 0.0;
            for (int n=0;n<N;++n) acc += (double)dLdz[n*num_classes + o] * (double)h_fc_in[n*fc_in_dim + d];
            float grad = (float)(acc / (double)N);
            // update host weight
            h_fc_w[o*fc_in_dim + d] -= lr * grad;
        }
        // bias grad
        double accb = 0.0;
        for (int n=0;n<N;++n) accb += dLdz[n*num_classes + o];
        float gradb = (float)(accb / (double)N);
        h_fc_b[o] -= lr * gradb;
    }
    // copy updated fc weights back to device
    checkCuda(cudaMemcpy(d_fc_w, h_fc_w, sizeof(float) * fc_w_size, cudaMemcpyHostToDevice), "memcpy fc_w H2D");
    checkCuda(cudaMemcpy(d_fc_b, h_fc_b, sizeof(float) * num_classes, cudaMemcpyHostToDevice), "memcpy fc_b H2D");

    printf("Performed simple host-side SGD update on FC weights (demo).\n");

    // Clean up and exit
    cudaFree(d_input);
    cudaFree(d_conv1_w); cudaFree(d_conv1_b); cudaFree(d_conv1_out);
    cudaFree(d_pool1_out);
    if (option==2) { cudaFree(d_conv2_w); cudaFree(d_conv2_b); cudaFree(d_conv2_out); cudaFree(d_pool2_out); }
    cudaFree(d_fc_in); cudaFree(d_fc_w); cudaFree(d_fc_b); cudaFree(d_logits);

    free(h_input); free(h_labels); free(h_logits); free(h_probs); free(h_fc_in);
    free(h_conv1_w); free(h_conv1_b); if (h_conv2_w) free(h_conv2_w); if (h_conv2_b) free(h_conv2_b);
    free(h_fc_w); free(h_fc_b); free(dLdz);

    printf("Done.\n");
    return 0;
}
