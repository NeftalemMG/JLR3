%%writefile mnist_loader.cu

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

uint32_t read_u32_be(FILE *f){
    unsigned char b[4];
    fread(b, 1, 4, f);
    return (b[0]<<24) | (b[1]<<16) | (b[2]<<8) | b[3];
}

unsigned char* load_idx_images(const char *path, int *outN, int *outH, int *outW){
    FILE *f = fopen(path,"rb");
    
    if (!f){ 
        fprintf(stderr,"Failed to open %s\n",path); 
        return NULL; 
    }
    
    read_u32_be(f);
    uint32_t N = read_u32_be(f);
    uint32_t H = read_u32_be(f);
    uint32_t W = read_u32_be(f);

    unsigned char *buf = (unsigned char*)malloc(N * H *W);
    fread(buf, 1, (N * H * W), f);
    fclose(f);

    *outN = N; 
    *outH = H; 
    *outW = W;

    return buf;
}

unsigned char* load_idx_labels(const char *path, int *outN){
    FILE *f = fopen(path,"rb");

    if (!f){
        fprintf(stderr,"Failed to open %s\n",path); 
        return NULL; 
    }

    read_u32_be(f);
    uint32_t N = read_u32_be(f);

    unsigned char *buf = (unsigned char*)malloc(N);
    fread(buf, 1, N, f);
    fclose(f);

    *outN = N;
    return buf;
}

void convert_images_u8_to_f32(unsigned char *u8, float *out, int N, int H, int W){
    for(int i = 0; i < (N * H * W); i++){
        out[i] = (((float)u8[i] / 255.0f) - 0.1307f) / 0.3081f;
    }
}