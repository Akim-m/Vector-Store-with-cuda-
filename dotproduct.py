import ctypes
import numpy as np

cuda_lib = ctypes.CDLL('./libdot_product_cublas.so')  


cuda_lib.calculate_dot_product.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]

def ensure_float32(array):
    
    if array.dtype != np.float32:
        return array.astype(np.float32)
    return array

def dotCuda(a, b):
    
    a = ensure_float32(a)
    b = ensure_float32(b)
    
    n = a.size

    result = np.zeros(1, dtype=np.float32)

    cuda_lib.calculate_dot_product(a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                   b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                   result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                   n)

    return result[0]

if __name__ == "__main__":

    a = np.random.rand(1000)
    b = np.random.rand(1000)

    result = dotCuda(a, b)
    print(f"Dot product : {result}")
