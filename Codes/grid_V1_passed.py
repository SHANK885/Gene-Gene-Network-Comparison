from pycuda import driver, gpuarray, tools
from pycuda.compiler import SourceModule
import pycuda.autoinit
import numpy as np
import pandas as pd
import timeit

start = timeit.default_timer()
# Source Module
mod = SourceModule("""
                __global__ void GraphComp(float *a, float *b, int *c, int size, float tvalue)
                {
                        const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
                        const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
                        //const unsigned int tid = idx + (idy * (blockDim.x * gridDim.x));

                        int Pvalue = 0;
                        float Avalue,Bvalue;

                        for(int k=0; k<size; k++)
                        {
                                Avalue = a[idx + (k * (blockDim.x * gridDim.x))];
                                Bvalue = b[idx + (k * (blockDim.x * gridDim.x))];
                        if((Avalue >= tvalue && Bvalue < tvalue) ||
                        	(Avalue < tvalue && Bvalue >= tvalue))
                        {
                                Pvalue += 1;
                        }
                        }
                        c[idx] = Pvalue;
                }""")

# Read Gene Matrix Dataset
a = pd.read_csv('../Datasets/good_data_2500.csv', header = None)
b = pd.read_csv('../Datasets/bad_data_2500.csv', header = None)

# Convert to matrix from dataframe
A = (a.iloc[:, :].values).astype(np.float32)
B = (b.iloc[:, :].values).astype(np.float32)

# Number of Nodes in Matrix
MatSize = len(A)
MATRIX_SIZE = np.int32(MatSize)

# set node expression value threshold
tvalue = 0.5
thres_value = np.float32(tvalue)

# transfer host memory to device memory
a_gpu = gpuarray.to_gpu(A)
b_gpu = gpuarray.to_gpu(B)

# copy matrix values form host variable to gpu variable
c_gpu = gpuarray.empty((MATRIX_SIZE), np.int32)

Graph = mod.get_function("GraphComp")

block_X = int(500)
block_Y = int(1)
grid_X = int(MATRIX_SIZE/500)
grid_Y = int(MATRIX_SIZE)

Graph(a_gpu, b_gpu, c_gpu, MATRIX_SIZE, thres_value,
		block = (block_X, block_Y, 1),
		grid = (grid_X, grid_Y))

# print values
print(c_gpu.get())
stop = timeit.default_timer()
print("Time:",stop-start)
