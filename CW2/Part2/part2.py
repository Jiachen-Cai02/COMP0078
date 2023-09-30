import numpy as np
from part2_functions import *

l_list = [1, 2, 4, 8, 16]

data_50  = np.loadtxt('dtrain13_50.dat')
data_100 = np.loadtxt('dtrain13_100.dat')
data_200 = np.loadtxt('dtrain13_200.dat')
data_400 = np.loadtxt('dtrain13_400.dat')




LI_50 = []
LKI_50 = []

##for data_50

for l in l_list:
    LI = []
    LKI = []
    for run in range(20):

        L = random_sample(data_50, l).astype(int)

        errLI = Laplacian_Interpolation(data_50, L)     
        errLKI = LaplacianKernelInterpolation_cjc(data_50, L)

        LI.append(errLI)
        LKI.append(errLKI)
    LI_50.append([np.mean(LI), np.std(LI)])
    LKI_50.append([np.mean(errLKI), np.std(LKI)])


LI_100 = []
LKI_100 = []

##for data_100


for l in l_list:
    LI = []
    LKI = []
    for run in range(20):

        L = random_sample(data_100, l).astype(int)

        errLI = Laplacian_Interpolation(data_100, L)     
        errLKI = LaplacianKernelInterpolation_cjc(data_100, L)

        LI.append(errLI)
        LKI.append(errLKI)
    LI_100.append([np.mean(LI), np.std(LI)])
    LKI_100.append([np.mean(errLKI), np.std(LKI)])


LI_200 = []
LKI_200 = []

##for data_200

for l in l_list:
    LI = []
    LKI = []
    for run in range(20):

        L = random_sample(data_200, l).astype(int)

        errLI = Laplacian_Interpolation(data_200, L)     
        errLKI = LaplacianKernelInterpolation_cjc(data_200, L)

        LI.append(errLI)
        LKI.append(errLKI)
    LI_200.append([np.mean(LI), np.std(LI)])
    LKI_200.append([np.mean(errLKI), np.std(LKI)])


LI_400 = []
LKI_400 = []

##for data_400

for l in l_list:
    LI = []
    LKI = []
    for run in range(20):

        L = random_sample(data_400, l).astype(int)

        errLI = Laplacian_Interpolation(data_400, L)     
        errLKI = LaplacianKernelInterpolation_cjc(data_400, L)

        LI.append(errLI)
        LKI.append(errLKI)
    LI_400.append([np.mean(LI), np.std(LI)])
    LKI_400.append([np.mean(errLKI), np.std(LKI)])


print('Errors and standard deviations for semi-supervised learning via Laplacian interpolation')
print('data points per label : 50')
for i in range(len(l_list)):
    print(LI_50[i]) 
  
print('data points per label : 100')
for i in range(len(l_list)):
    print(LI_100[i]) 
    
print('data points per label : 200')
for i in range(len(l_list)):
    print(LI_200[i]) 

print('data points per label : 400')
for i in range(len(l_list)):
    print(LI_400[i]) 
    
print('Errors and standard deviations for semi-supervised learning via Laplacian kernel interpolation')
print('data points per label : 50')
for i in range(len(l_list)):
    print(LKI_50[i]) 
  
print('data points per label : 100')
for i in range(len(l_list)):
    print(LKI_100[i]) 
    
print('data points per label : 200')
for i in range(len(l_list)):
    print(LKI_200[i]) 

print('data points per label : 400')
for i in range(len(l_list)):
    print(LKI_400[i]) 




