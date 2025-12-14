**#ifndef, #define, #endif** => This is called an **include guard** and it prevents code from being included multiple times. 

So we are just basically saying that if a certain code block has not been defined yet, (ifndef), then define it now (define), and after finishing up, you close it (endif). 

**fread** => this is a binary input function that reads raw bytes from a file into memory. This is its function signature:  
  
size_t fread(void *ptr, size_t size, size_t count, FILE *stream);
  
```
The parameters: 1) ptr: pointer to where the data will be stored in memory
                2) size: size in bytes of one item
                3) count: number of items to read
                4) stream: pointer to the file we are going to be reading from. The file must be opened in binary form ("rb")
                Ex: File *f = fopen("whateverYourFileNameIs.bin", "rb");
```


**macros** => macros in c are not functions or not code that run - they simply are just text substitution rule handled by the preprocessor.  
  
Before the compiler even sees our C code, the preprocessor reads our file, applies macros, produces a new expanded source file and then that gets compiled.  

Macros basically are "Find this text and replace it with that text."  
  
Ex: #define PI 3.14159.

This means that every time we see PI, it will be replaced with 3.14159. 

```
Macros with arguments (function-like macros) => Ex: #define SQUARE(x) ((x) * (x)) 
                                                    int a = SQUARE(1)


However, just keep in mind that macros can be scary because 1) They ignore types
                                                            2) Ignore Scopes
                                                            3) Can evaluate arguments multiple times
                                                            4) Not debuggable
```
But why even use macros at all?
It is because they can do things that functions cannot:
```
1) Compile time constraints. Ex: #define IMG_H 28
                                 #define IMG_W 28
There is no runtime or memory cost. 

2) Conditional Compilation. Ex: #define DEBUG
                                #ifdef DEBUG
                                    printf("x = %d\n", x);
                                #endif DEBUG
If DEBUG is defined, code exists, if not, code is removed entirely. No runtime if. 

3) Platform specific code. Ex: #ifdef WIN32
                                    // Windows code
                               #else
                                    // Linux/Mac code
                               #endif

4) Code generation (dangerous (for the reasons we mentioned earlier but still powerful))
Ex: #define FOR_EACH_PIXEL(i, N) for(int i = 0; i < (N); i++)

Used as: 
FOR_EACH_PIXEL(i, N) {
    out[i] = ...
}
```
**do { ... } while (0)** => This pattern creates a single statement with its own scope and executes only once. 

To understand the need for this pattern, lets see an example where an error occurs because the pattern was not applied:
```
Lets imagine this macro:
#define CHECK_CUDA(call) \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("CUDA error\n"); \
        exit(1); \
    }

Now lets use it here:
if (ok)
    CHECK_CUDA(cudaMalloc(&ptr, size));
else
    cleanup();

After the macro expansion, the compiler sees:
if (ok)
    cudaError_t error = cudaMalloc(&ptr, size);
if (error != cudaSuccess) {
    printf("CUDA error\n");
    exit(1);
}
else
    cleanup();
```
And Boom, we have a syntax and a logic error:

In our case, else will bind to the wron if and the macro has expanded into multiple statements. C has no idea what you mean, hence the errors. 
