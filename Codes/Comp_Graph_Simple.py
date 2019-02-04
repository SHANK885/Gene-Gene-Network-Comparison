import numpy as np
from pycuda import driver, gpuarray, tools
import pandas as pd
import pycuda.autoinit
from pycuda.compiler import SourceModule

# Source Module
mod = SourceModule("""
		__global__ void GraphComp(float *a, float *b, int *c, int size, float tvalue){
			int tx = threadIdx.x;
			int Pvalue = 0;
 			float Avalue,Bvalue;
			for(int k=0; k<size; k++){
				Avalue = a[tx + size * k];
				Bvalue = b[tx + size * k];
       				if((Avalue >= tvalue && Bvalue < tvalue) || (Avalue < tvalue && Bvalue >= tvalue)){       
        				Pvalue += 1;
        			}
			}
			c[tx] = Pvalue;
		}""")

# Read Gene Matrix Datasets
a = pd.read_csv('Dataset/a_data_33.csv', header = None)
b = pd.read_csv('Dataset/b_data_33.csv', header = None)

# Convert to matrix form dataframe
A = (a.iloc[:,:].values).astype(np.float32)
B = (b.iloc[:,:].values).astype(np.float32)

# Number of Nodes in Matrix
MatSize = len(A)
MATRIX_SIZE = np.int32(MatSize)

# set node expression value threshold
tvalue = 0.5
thres_value = np.float32(tvalue)

# transfer host memory to device memory
a_gpu = gpuarray.to_gpu(A)
b_gpu = gpuarray.to_gpu(B)

# copy matrix values form host variable to gpu variablE
c_gpu = gpuarray.empty((MATRIX_SIZE), np.int32)

Graph = mod.get_function("GraphComp")

Graph(a_gpu, b_gpu, c_gpu, MATRIX_SIZE, thres_value, block = (MatSize, MatSize, 1))

# print values
print('\n')
print(a)
print('\n')
print(b)
print('\n')
print(c_gpu.get())
