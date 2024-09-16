import ctypes
import numpy as np

cuda_lib = ctypes.CDLL('./libcuda_norm_cublas.so') 


cuda_lib.cuda_norm_cublas.restype = ctypes.c_float
cuda_lib.cuda_norm_cublas.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]

def normCu(vector):
    n = len(vector)
    vector_ctypes = (ctypes.c_float * n)(*vector)

    # Call the CUDA function
    result = cuda_lib.cuda_norm_cublas(vector_ctypes, n)
    
    return result

if __name__ == "__main__":
    vector = np.random.rand(1000).astype(np.float32)

    norm = normCu(vector)
    print(f"Optimized cuBLAS norm of the vector: {norm}")
