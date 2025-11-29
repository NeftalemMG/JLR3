%%writefile triton_fix.py

"""
Triton libcuda.so fix for Kaggle
Import this BEFORE any other Triton imports
"""
import os
import sys

# Setup paths
os.environ['TRITON_LIBCUDA_PATH'] = '/tmp/cuda_lib'
os.environ['LD_LIBRARY_PATH'] = '/tmp/cuda_lib:/usr/local/nvidia/lib64:/usr/local/cuda-12.5/compat:' + os.environ.get('LD_LIBRARY_PATH', '')

# Create symlink
os.makedirs('/tmp/cuda_lib', exist_ok=True)
if not os.path.exists('/tmp/cuda_lib/libcuda.so'):
    try:
        os.symlink('/usr/local/nvidia/lib64/libcuda.so.1', '/tmp/cuda_lib/libcuda.so')
        print("Created libcuda.so symlink at /tmp/cuda_lib/")
    except FileExistsError:
        pass

# Monkey patch Triton
try:
    import triton.common.build
    
    def patched_libcuda_dirs():
        """Return our custom libcuda directory"""
        cuda_dirs = ['/tmp/cuda_lib']
        # Verify the directory exists and has libcuda.so
        if os.path.exists('/tmp/cuda_lib/libcuda.so'):
            return cuda_dirs
        else:
            raise RuntimeError("libcuda.so not found in /tmp/cuda_lib/")
    
    # Replace the function
    triton.common.build.libcuda_dirs = patched_libcuda_dirs
    
    print("Patched triton.common.build.libcuda_dirs")
    print(f"libcuda.so confirmed at: /tmp/cuda_lib/libcuda.so")
    
except ImportError:
    print(" Triton not imported yet, patch will apply when imported")

print("Triton fix applied!\n")