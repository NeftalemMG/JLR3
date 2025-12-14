**CUDA** => Compute Unified Device Architecture

**Kernel** => A function that runs on the GPU Thread

**Block** => A group of threads (they can cooperate)

**Grid** => Collection of all blocks

**-----------------------------------------------------------------------**

### C / C++/ CUDA Related Stuff
Even though most of our files look like C, .cu files are compiled as C++, not C. 

So we are not limited to functions or structs only, which is the case in C, but we also get
access to RAII, Constructors/Destructors, Encapsulation, and Methods. 

**RAII** => Resource Acquistion Is Intialization

This basically means that if an object exists, the resource is valid. However, when the objects stops existing, the resource is automatically cleaned up. 

Why does RAII Exist? (What Problem Does it Solve?)

In C-style code, we usually do this:
```
FILE *f = fopen("data.txt", "r");
/* use file */
fclose(f);
```
The problem is what happens if you return early or an error happens or someone forgets to fclose?
We get a resouce leak. 

If you imagine that with GPU memory, CUDA events etc, it can be problematic. 

RAII basically in one sentence: "Tie the lifetime of a resource to the lifetime of an object."

The RAII pattern (core IDEA):

A class does three things:
1) Constructor => Acquires the resource
2) Destructor => Release the resource
3) Object lifetime => Guarantees correctness

Cuda example from cudautils.cu (perfect RAII):
```cpp
class CudaTimer {
public:
    CudaTimer() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }

    ~CudaTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
};
```

This guarantees that events are created and destroyed only once, and clean up is not forgotten.

How RAII behaves in real code:
```cpp
void run() {
    CudaTimer timer;  // constructor runs here
    timer.start();

    if (error) return;  // destructor STILL runs

    timer.stop();
}  // destructor runs here too
```

This ensures that there are no leaks, no special cleanup logic, and no goto cleanup hacks (referring to C here).

**Constructors** => A constructor is a special function that runs automatically when an object is created. Its job is to set the object up so it starts life in a valid state. We never call it ourselves. 
```cpp
class Person {
public:
    int age;

    Person() {
        age = 0;
    }
};
```
usage: Person p; //constructor runs here

After creation, p.age == 0

**Destructors** => A destructor is a special function that runs automatically when an object is destroyed. Its job is to clean up resources before the object dies. Again, we never call it ourselves. 
```cpp
class Person {
public:
    ~Person() {
        printf("Person destroyed\n");
    }
};
```
When does it run?
```cpp
{
    Person p;
}   // destructor runs here
```
Syntax:
Constructor	ClassName(...)
Destructor	~ClassName()
```cpp
class A {
public:
    A() {}      // constructor
    ~A() {}     // destructor
};
```
Without destructors, there is no clean up, and as a result, memory leaks. 

This ties with RAII because as per the RAII rules, constructor acquires the resource and destructor releases resource. 