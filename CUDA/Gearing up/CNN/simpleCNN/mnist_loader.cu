%%writefile mnist_loader.cu

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// We are going to be reading 4-byte big endian numbers
// The MNIST files store integers in big-endian format, so we need to convert them.
// I had no idea what that meant, so here is the explanation:
// Mnist images and and labels come in this format called IDX, as one will see in the downloaded dataset.
// IDX is a special binary format. So if we take our training dataset, train-images.idx3-ubyte, as an example, 
// The first 16 bytes of train-images.idx3-ubyte looks like this:

// Bytes 0 -> 3: This is the magic number and it is 2051 for images (It basically tells our program, "hey, this is an MNIST image file" (or label file))

// Bytes 4 -> 7: These bytes specify the number of images that are going to be in our dataset. That number is written in binary format.
// That could be something like 00 00 EA 60, which is read as 0x0000EA60, and that is the hex equivalent of the decimal 60000. 
// Which means we have 60000 images in that training set.

// Bytes 8 -> 11: These bytes specify the number of rows (Number of horizontal lines of pixels each image has)

// Bytes 12 -> 15: These bytes specify the number of columns (Number of vertical lines of pixels each image has)
// Since MNIST images are 28x28 pixels, this means that we have 28 rows and 28 columns.

// Going back to the meaning of 4-byte big-endian number, 
// 4 bytes as we know is 32 bits, 
// And big Endian basically means the most significant byte comes first.
// So in our case, as we mentioned earlier, the first 16 bytes (aka the "header" as a whole) are the significant part that specify all the details
// about what data is stored in the dataset.
// Then, bytes 16 onwards contain all the image pixel data.
// To be more specific, 
// Each pixel is 1 byte (8 bits) in the range [0, 255], 0 meaning black and 255 meaning white, while everything in between is shades of grey
// Pixels are stored sequentially, row by row, so the first 28 bytes are pixels of row 0, next 28 bytes are pixels of row 1 and the last 28 bytes of that image are pixels of row 27. 
// So for one 28 x 28 image, we have 784 bytes.

// Let say we had a 4 x 4 image:
// 0   0   0   0 => Row 0
// 0 255 255   0 => Row 1
// 0 255 255   0 -> Row 2
// 0   0   0   0 => Row 3

// In our file, they would be stored like this (decimal values as bytes): 0,0,0,0,0,255,255,0,0,255,255,0,0,0,0,0 


uint32_t read_u32_be(FILE *f){
    // uint32_t => unsigned 32 bit int
    // read_u32_be => read unsigned 32 bit big endian
    unsigned char b[4];
    fread(b, 1, 4, f); // read 4 bytes from the file f and store them in b
    return (b[0]<<24) | (b[1]<<16) | (b[2]<<8) | b[3];
    // Shifts b[0] 24 bits to the left, so that it becomes the most significant byte (the magic number specifying byte)
    // b[1] and b[2] are shifted 16 and 8 bytes to the left, respectively, while b[3] (the number of columns) byte stays where it is.
}


// Load MNIST images
// path: file path to MNIST images
// outN, outH, outW: number of images, height, width
// returns pointer to uint8 image array (0-255)
unsigned char* load_idx_images(const char *path, int *outN, int *outH, int *outW){
    FILE *f = fopen(path,"rb"); // open file for reading in binary mode
    
    if (!f){ 
        fprintf(stderr,"Failed to open %s\n",path); 
        return NULL; 
    }
    
    read_u32_be(f); // skip magic number
    uint32_t N = read_u32_be(f); // number of images
    uint32_t H = read_u32_be(f); // image height
    uint32_t W = read_u32_be(f); // image width

    // Another thing that was a little confusing to me was why the reading operation was not automatically starttng at index 0 every time we call it.
    // However, fread does NOT restart at the beginning every time. The file pointer automatically moves forward. In C, when you open a file:
    // FILE *f = fopen(...); the file has an internal cursor called the file position indicator. Every time we read something, the bookmark moves forward.
    // Therefore, after the read operaions above, the file pointer is now at byte 16, right where the pixel data begins.

    unsigned char *buf = (unsigned char*)malloc(N * H *W); // allocate buffer
    fread(buf, 1, (N * H * W), f); // read all image data
    fclose(f);

    *outN = N; 
    *outH = H; 
    *outW = W;

    return buf;
}

// Loading the MNIST Labels
// path: file path to MNIST labels
// outN: number of labels
// returns pointer to uint8 labels
unsigned char* load_idx_labels(const char *path, int *outN){
    FILE *f = fopen(path,"rb");

    if (!f){
        fprintf(stderr,"Failed to open %s\n",path); 
        return NULL; 
    }

    read_u32_be(f); // skipping the magic number again
    uint32_t N = read_u32_be(f);

    unsigned char *buf = (unsigned char*)malloc(N);
    fread(buf, 1, N, f);
    fclose(f);

    *outN = N;
    return buf;
}

// Converting images from uint8 to float and normalizing them
// u8: input uint8 images
// out: output float array
// N, H, W: number of images, height, width
void convert_images_u8_to_f32(unsigned char *u8, float *out, int N, int H, int W){
    for(int i = 0; i < (N * H * W); i++){
        // normalize pixels to mean = 0, standard deviation = 1 roughly like PyTorch MNIST
        // But why are we even doing this?
        // Subtracting the mean re-centers the data at 0. Ex: 5, 6, 7 => Mean = 6, Subtract the mean ⇒ -1, 0, +1
        // Now the center is 0, not 6.
        // Why divide by the standard deviation?
        // Lets say these are our input: -100, 0, 100. These are HUGE. NNs hate big inputs; gradients explode, weights blow up.
        // Dividing by standard deviation scales the data so it's not too spread out. Std dev = 81.65
        // So when we divide them, we have: -1.22, 0, +1.22
        // NNs work BEST when inputs have mean ≈ 0, have spread ≈ 1 (standard deviation ≈ 1)
        // This gives: stable gradients, faster learning, better convergence, fewer training issues

        // The mean of our dataset is roughly 0.1307 and the standard deviation is roughly 0.3081

        out[i] = (((float)u8[i] / 255.0f) - 0.1307f) / 0.3081f;
    }
}
