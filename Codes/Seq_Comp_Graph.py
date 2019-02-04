import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule
import pandas as pd
import timeit

start = timeit.default_timer()

# Read Gene Matrix Datasets
a = pd.read_csv('Dataset/a_data_10000.csv', header = None)
b = pd.read_csv('Dataset/b_data_10000.csv', header = None)

# Convert to matrix form dataframe
A = (a.iloc[:,:].values).astype(np.float32)
B = (b.iloc[:,:].values).astype(np.float32)

# Number of Nodes in Matrix
MatSize = len(A)

# set node expression value threshold
tvalue = 0.5

count = [0 for i in range(0,MatSize)]

for i in range(0,MatSize):
	for j in range(0,MatSize):
		if (A[i][j]>=tvalue and B[i][j]<tvalue) or (A[i][j]<tvalue and B[i][j]>=tvalue):
			count[i]=count[i]+1

print(count)

stop = timeit.default_timer()
print("Time =",stop-start)
