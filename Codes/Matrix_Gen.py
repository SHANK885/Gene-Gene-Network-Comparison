import numpy as np
import random


n = int(input("Enter the number of nodes in Matrix: "))
good_data = np.random.random((n, n))

for row in range(0, n):
	for col in range(row, n): 
		if row == col:
			good_data[row][col] = 0
		else:
			good_data[col][row] = good_data[row][col]

bad_data = good_data.copy()

for row in range(0, n):
    	for col in range(row+1, n):
        	rand_num = random.random()
        	if(rand_num <= .4) :
            		bad_data[row][col] = rand_num
            		bad_data[col][row] = rand_num

print(good_data)
print(bad_data)

np.savetxt("Dataset/a_data_{}.csv".format(n), good_data, fmt='%.2f', delimiter= ",")
np.savetxt("Dataset/b_data_{}.csv".format(n), bad_data, fmt='%.2f', delimiter= ",")

print("Dataset Matrices of {} by {} created and saved to csv files".format(n, n))
