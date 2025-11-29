%%writefile main.cu

#include <stdio.h>
#include <stdlib.h>
#include "mnist_loader.cu"   // for loading MNIST
#include "kernels.cu"        // for conv2d, relu, maxpool, fc kernels
#include "utils.cu"          // for weight init, error checking, softmax+loss

// Main file for 
// loading the MNIST dataset, normalizing images, setting up model weights, sending everything to GPU
// running  a simple Conv => ReLU => MaxPool => Fully Connected pipeline
// and computing softmax + cross entropy loss on CPU


int main() {

    srand(1234); // So that our weight initialization is reproducible
    // If I am not mistaken, this is the the Cuda/C equivalent of numpy's seed function.

    // 1) LOAD MNIST DATASET

    int totalTrainImages, imageH, imageW;

    // This loads the raw MNIST ubyte file.
    // train-images.idx3-ubyte contains ALL the training images (60k)
    unsigned char* trainImagesU8 =
        load_idx_images("kaggle/input/mnist-dataset/train-images.idx3-ubyte",
                        &totalTrainImages, &imageH, &imageW);

    int totalTrainLabels;
    unsigned char* trainLabels =
        load_idx_labels("/kaggle/input/mnist-dataset/train-labels.idx1-ubyte",
                        &totalTrainLabels);

    if (!trainImagesU8) {
        printf("Failed to load MNIST images. Exiting.\n");
        return 1;
    }

    printf("Loaded MNIST: %d images of size %dx%d\n",
           totalTrainImages, imageH, imageW);

    // We will NOT train on 60k images for now.
    // Instead, we will take a tiny subset so we can test the pipeline end-to-end.
    int N = 64;  // number of samples we will use

    // Allocate floating-point buffer for normalized images
    float* hostInput = (float*)malloc(sizeof(float) * N * imageH * imageW);

    // Convert raw images from uint8 => float32 and normalize
    convert_images_u8_to_f32(trainImagesU8, hostInput, N, imageH, imageW);

    // Copy labels for these first N samples
    unsigned char* hostLabels = (unsigned char*)malloc(N);
    for (int i = 0; i < N; i++) {
        hostLabels[i] = trainLabels[i];
    }

    // Free the original big MNIST buffer because we no longer need it
    free(trainImagesU8);
    free(trainLabels);


    // Model architecture Setup

    // This is a very small CNN:
    // Conv layer:    1 => 16 filters
    // Kernel size:   3×3
    // Padding:       1 (same padding)
    // Stride:        1

    int numInputChannels = 1;     // MNIST is grayscale
    int numConvFilters = 16;      // Number of output channels after convolution
    int kernelSize = 3;
    int padding = 1;
    int stride = 1;

    // Output size formula for convolution with padding:
    // H_out = (H + (2 * pad) - K)/stride + 1
    int convH = (imageH + 2 * padding - kernelSize) / stride + 1;
    int convW = (imageW + 2 * padding - kernelSize) / stride + 1;

    // Then we apply maxpool 2×2, shrinking the size by half
    int poolH = convH / 2;
    int poolW = convW / 2;

    // The fully connected layer takes the flattened pooled output
    int fcInputDim = numConvFilters * poolH * poolW;
    int numClasses = 10; // digits 0-9

    // Allocate weights on CPU
    // In CNNs, filters are literally the weights
    // Each filter is going to be a small array of numbers that the convolution uses to detect patterns.
    // Filters are not hardcoded with specific values, like "edge detector" or "line detector" - they are randomly initialized at first
    // During training, these filter values will be adjusted to detect useful features

    // Here,  
    // numConvFilters = C_out, the number of filters you want
    // numInputChannels = C_in, number of channels in input
    // kernelSize × kernelSize = spatial size of each filter
    float* hostConvWeights =
        (float*)malloc(sizeof(float) * numConvFilters * numInputChannels * kernelSize * kernelSize);

    float* hostConvBias =
        (float*)malloc(sizeof(float) * numConvFilters);

    float* hostFcWeights =
        (float*)malloc(sizeof(float) * numClasses * fcInputDim);

    float* hostFcBias =
        (float*)malloc(sizeof(float) * numClasses);

    // Initialize weights randomly (small values) and bias to 0
    init_weights(hostConvWeights,
                 numConvFilters * numInputChannels * kernelSize * kernelSize);

    zero_fill(hostConvBias, numConvFilters);

    init_weights(hostFcWeights, numClasses * fcInputDim);
    zero_fill(hostFcBias, numClasses);

    //Allocate Device Memory + Copy Weights and Input to GPU
    float *d_input, *d_convW, *d_convB, *d_convOut;
    float *d_poolOut, *d_fcIn, *d_fcW, *d_fcB, *d_logits;

    // Input images (N × H × W)
    size_t inputBytes = sizeof(float) * N * imageH * imageW;
    checkCuda(cudaMalloc(&d_input, inputBytes), "Alloc d_input");
    checkCuda(cudaMemcpy(d_input, hostInput, inputBytes, cudaMemcpyHostToDevice),
              "Copy hostInput to d_input");

    // Convolution weights
    size_t convWBytes = sizeof(float) * numConvFilters * numInputChannels * kernelSize * kernelSize;
    checkCuda(cudaMalloc(&d_convW, convWBytes), "Alloc d_convW");
    checkCuda(cudaMemcpy(d_convW, hostConvWeights, convWBytes, cudaMemcpyHostToDevice),
              "Copy conv weights");

    // Convolution bias
    size_t convBBytes = sizeof(float) * numConvFilters;
    checkCuda(cudaMalloc(&d_convB, convBBytes), "Alloc d_convB");
    checkCuda(cudaMemcpy(d_convB, hostConvBias, convBBytes, cudaMemcpyHostToDevice),
              "Copy conv bias");

    // Output of convolution layer
    size_t convOutBytes = sizeof(float) * N * numConvFilters * convH * convW;
    checkCuda(cudaMalloc(&d_convOut, convOutBytes), "Alloc d_convOut");

    // Maxpool output
    size_t poolOutBytes = sizeof(float) * N * numConvFilters * poolH * poolW;
    checkCuda(cudaMalloc(&d_poolOut, poolOutBytes), "Alloc d_poolOut");

    // FC input (flattened pooled output)
    size_t fcInBytes = sizeof(float) * N * fcInputDim;
    checkCuda(cudaMalloc(&d_fcIn, fcInBytes), "Alloc d_fcIn");

    // FC weights
    size_t fcWBytes = sizeof(float) * numClasses * fcInputDim;
    checkCuda(cudaMalloc(&d_fcW, fcWBytes), "Alloc d_fcW");
    checkCuda(cudaMemcpy(d_fcW, hostFcWeights, fcWBytes, cudaMemcpyHostToDevice),
              "Copy FC weights");

    // FC bias
    size_t fcBBytes = sizeof(float) * numClasses;
    checkCuda(cudaMalloc(&d_fcB, fcBBytes), "Alloc d_fcB");
    checkCuda(cudaMemcpy(d_fcB, hostFcBias, fcBBytes, cudaMemcpyHostToDevice),
              "Copy FC bias");

    // FC output logits
    size_t logitsBytes = sizeof(float) * N * numClasses;
    checkCuda(cudaMalloc(&d_logits, logitsBytes), "Alloc d_logits");


    // Forward pass loop (For now, just doing 2 epchs to show output)
    // Bascally useless at this point

    int threads = 256;

    for (int epoch = 0; epoch < 2; epoch++) {
        // Convolution Layer

        int totalConvOutputs = N * numConvFilters * convH * convW;
        int convBlocks = (totalConvOutputs + threads - 1) / threads;

        conv2d_simple_kernel<<<convBlocks, threads>>>(
            d_input,
            d_convW,
            d_convB,
            d_convOut,
            N,
            numInputChannels,
            imageH,
            imageW,
            numConvFilters,
            kernelSize,
            padding,
            stride,
            convH,
            convW
        );
        cudaDeviceSynchronize();

        // ReLU Activation-
        relu_kernel<<<convBlocks, threads>>>(d_convOut, d_convOut, totalConvOutputs);
        cudaDeviceSynchronize();

        // MaxPool 2×2
        int totalPoolOutputs = N * numConvFilters * poolH * poolW;
        int poolBlocks = (totalPoolOutputs + threads - 1) / threads;

        maxpool2x2_kernel<<<poolBlocks, threads>>>(
            d_convOut,
            d_poolOut,
            N,
            numConvFilters,
            convH,
            convW,
            poolH,
            poolW
        );
        cudaDeviceSynchronize();

        // Flatten pooled output => fully connected input
        for (int sample = 0; sample < N; sample++) {
            checkCuda(
                cudaMemcpy(
                    d_fcIn + sample * fcInputDim,
                    d_poolOut + sample * numConvFilters * poolH * poolW,
                    sizeof(float) * numConvFilters * poolH * poolW,
                    cudaMemcpyDeviceToDevice),
                "Flatten step"
            );
        }

        // Fully Connected Forward

        int totalFcOutputs = N * numClasses;
        int fcBlocks = (totalFcOutputs + threads - 1) / threads;

        fc_forward_kernel<<<fcBlocks, threads>>>(
            d_fcIn,
            d_fcW,
            d_fcB,
            d_logits,
            N,
            fcInputDim,
            numClasses
        );
        cudaDeviceSynchronize();

        // Copy logits back to host and copmute softmax and loss
        float* hostLogits = (float*)malloc(logitsBytes);
        float* hostSoftmaxProbs = (float*)malloc(logitsBytes);
        float loss = 0;

        checkCuda(cudaMemcpy(hostLogits, d_logits, logitsBytes, cudaMemcpyDeviceToHost),
                  "Copy logits");

        // Compute softmax on CPU for simplicity
        softmax_and_loss_host(hostLogits, hostLabels, N, numClasses,
                              hostSoftmaxProbs, &loss);

        printf("\nEPOCH %d\n", epoch);
        printf("Loss = %f\n", loss);

        // Compute Accuracy + Print Predictions

        int correctCount = 0;

        for (int i = 0; i < N; i++) {
            // Find argmax probability
            int predictedDigit = 0;
            float bestProb = hostSoftmaxProbs[i * numClasses];

            for (int j = 1; j < numClasses; j++) {
                float p = hostSoftmaxProbs[i * numClasses + j];
                if (p > bestProb) {
                    bestProb = p;
                    predictedDigit = j;
                }
            }

            if (predictedDigit == hostLabels[i])
                correctCount++;

            printf("Sample %2d | True: %d | Pred: %d | Prob: %.5f\n",
                   i, hostLabels[i], predictedDigit, bestProb);
        }

        printf("Epoch %d accuracy: %.4f\n",
               epoch, (float)correctCount / N);

        free(hostLogits);
        free(hostSoftmaxProbs);
    }


    // Cleaning time
    // Free them all
    cudaFree(d_input);
    cudaFree(d_convW);
    cudaFree(d_convB);
    cudaFree(d_convOut);
    cudaFree(d_poolOut);
    cudaFree(d_fcIn);
    cudaFree(d_fcW);
    cudaFree(d_fcB);
    cudaFree(d_logits);

    free(hostInput);
    free(hostLabels);
    free(hostConvWeights);
    free(hostConvBias);
    free(hostFcWeights);
    free(hostFcBias);

    printf("\n;)!\n");
    return 27;
}
