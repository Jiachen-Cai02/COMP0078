#Part 1 for the CW1
#from pydoc import plain
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

####Q1 functions####

def transformed_dataset (xdata, k):
    # transformed the dataset in matrix notation

    result = np.zeros((len(xdata), k))
    for i in range(k):
        result [:,i] = xdata**i
    return result

def get_coeff (x, y, k):
    # get the coefficient for the linear regression
    x_matrix = transformed_dataset (x, k)

    # w = x-1y
    W = np.matmul(np.linalg.pinv(x_matrix), y)

    return W

def get_MSE (x, y, k):

    mapped_x = transformed_dataset(x, k)
    coe = get_coeff(x, y, k)
    SSE = (np.matmul(mapped_x, coe) - y)**2
    # MSE = mean of SSE
    MSE = np.sum(SSE) / x.shape[0]
    return MSE

####Q2 function ####

def gen_dataset(x, var, k):
    random_variable = np.random.normal(0, var, k)
    y = np.zeros(k)
    for i in range(len(x)):
        y[i] = (np.sin(2 * np.pi * x[i]))**2 + random_variable[i]
    return y

def gen_test_error(x_train, y_train, x_test, y_test, k):
    coe = get_coeff(x_train, y_train, k)
    mapped_test = transformed_dataset(x_test, k)
    SSE_test = (np.matmul(mapped_test, coe) - y_test)**2
    MSE_test = np.sum(SSE_test) / x_test.shape[0]
    return MSE_test

#### Q3 function ####

def transformed_dataset_Q3 (xdata, k):
    result = np.zeros((len(xdata), k))
    for i in range(1, k+1):
        result [:,i - 1] = np.sin(i * np.pi * xdata)
    return result

def get_coeff_Q3 (x, y, k):
    x_matrix = transformed_dataset_Q3 (x, k)
    W = np.matmul(np.linalg.pinv(x_matrix), y)
    return W

def get_MSE_Q3 (x, y, k):
    mapped_x = transformed_dataset_Q3(x, k)
    coe = get_coeff_Q3(x, y, k)
    SSE = (np.matmul(mapped_x, coe) - y)**2
    MSE = np.sum(SSE) / x.shape[0]
    return MSE

def gen_test_error_Q3(x_train, y_train, x_test, y_test, k):
    coe = get_coeff_Q3(x_train, y_train, k)
    mapped_test = transformed_dataset_Q3(x_test, k)
    SSE_test = (np.matmul(mapped_test, coe) - y_test)**2
    MSE_test = np.sum(SSE_test) / x_test.shape[0]
    return MSE_test

#####Q1(a)#####
x_origin = [1,2,3,4]
y_origin = [3,2,0,5]
array_x = np.array(x_origin)
array_y = np.array(y_origin)
k = 1
plt.figure(figsize=(8,6))
while k <= 4:
    x_data = transformed_dataset(array_x, k)
    x = np.linspace(0,5,100)
    coeff = get_coeff(array_x, array_y, k)
    y = np.matmul(transformed_dataset(x,k), coeff)
    plt.xlim(-0.5,5.5)
    plt.ylim(-5,8)
    plt.plot(x,y,  label = 'k = ' + str(k))
    plt.legend()
    k += 1
plt.scatter(array_x, array_y)
ax = plt.gca()
ax.spines['bottom'].set_position('zero')
plt.show()

#####Q1(b)#####
for i in range(1,5):
    print(get_coeff(array_x, array_y, i))

#####Q1(c)#####
for i in range(1,5):
    print (get_MSE(array_x, array_y, i))


#####Q2(a) i#####
data_set = np.random.uniform(0, 1, 30)
x2 = np.linspace(0,1,200)
y2 = (np.sin(2 * np.pi * x2))**2
plt.figure(figsize=(8, 6))
data_set_y = gen_dataset(data_set, 0.07, 30)
plt.scatter(data_set, data_set_y, color = 'black')
plt.plot(x2, y2, label = '$sin^2(2 \pi x)$')
plt.legend()
plt.show()

#####Q2(a) ii#####
pol_dim = [2, 5, 10, 14, 18]
for k in pol_dim:
    W = get_coeff(data_set, data_set_y, k)
    y = np.matmul(transformed_dataset(x2, k), W)
    plt.plot(x2, y, label = 'k =' + str(k))
plt.legend()
plt.xlim(0,1.2)
plt.ylim(-0.8, 1.5)
ax = plt.gca()
ax.spines['bottom'].set_position('zero')
plt.scatter(data_set, data_set_y, color = 'black')
plt.show()

#####Q2(b)#####
k = np.array(range(1,19))
ln_MSE = np.zeros(18)

for dim in k:
    ln_MSE[dim -1] = np.log(get_MSE(data_set, data_set_y, dim))


plt.figure(figsize = (8, 6))
plt.plot(k, ln_MSE, label = 'Log of training error')
plt.xlabel('dimension k', fontsize = 17)
plt.ylabel('Log(MSE) for training', fontsize = 17)
plt.xticks(k)
plt.legend()
plt.show()

#####Q2(c)#####
test_set_x = np.random.uniform(0, 1, 1000)
test_set_y = gen_dataset(test_set_x, 0.07, 1000)
k_test = np.array(range(1,19))
ln_MSE_test = np.zeros(18)
for dim in k_test:
    ln_MSE_test[dim - 1] = np.log(gen_test_error(data_set, data_set_y, test_set_x, test_set_y, dim))

plt.figure(figsize = (8, 6))
plt.plot(k_test, ln_MSE_test, label = 'Log of training error for test data')
plt.xticks(k_test)
plt.xlabel('dimension k', fontsize = 17)
plt.ylabel('Log(MSE) for testing', fontsize = 17)
plt.legend()
plt.show()


#####Q2(d)#####
MSE_train = []
MSE_test = []
for dim in k:
    MSE_sum_train = 0
    MSE_sum_test = 0
    for i in range(100):
        x_train = np.random.uniform(0, 1, 30)
        x_test = np.random.uniform(0, 1, 1000)
        y_train = gen_dataset(x_train, 0.07, 30)
        y_test = gen_dataset(x_test, 0.07, 1000)
        MSE_sum_train += get_MSE(x_train, y_train, dim)
        MSE_sum_test += gen_test_error(x_train, y_train, test_set_x, y_test, dim)
    MSE_train.append(np.log(MSE_sum_train / 100))
    MSE_test.append(np.log(MSE_sum_test / 100))


plt.figure(figsize = (8, 6))
plt.plot(k, MSE_train, label = 'log avg of training error for 100 times')
plt.plot(k_test, MSE_test, label = 'log avg of testing error for 100 times')
plt.xlabel('dimension k', fontsize = 17)
plt.ylabel('Log(avg MSE)', fontsize = 17)
plt.xticks(k)
plt.legend()
plt.show()

#####Q3 (b)#####
data_set = np.random.uniform(0, 1, 30)
data_set_y = gen_dataset(data_set, 0.07, 30)
k_Q3 = np.array(range(1,19))
ln_MSE_Q3 = np.zeros(18)
for dim in k:
    ln_MSE_Q3[dim -1] = np.log(get_MSE_Q3(data_set, data_set_y, dim))
plt.figure(figsize = (8, 6))
plt.plot(k_Q3, ln_MSE_Q3, label = 'Log of training error of new basis')
plt.xlabel('dimension k', fontsize = 17)
plt.ylabel('Log(MSE)', fontsize = 17)
plt.xticks(k)
plt.legend()
plt.show()

#####Q3 (c) #####
test_set_x_Q3 = np.random.uniform(0, 1, 1000)
test_set_y_Q3 = gen_dataset(test_set_x_Q3, 0.07, 1000)
ln_MSE_test_Q3 = np.zeros(18)
for dim in k_test:
    ln_MSE_test_Q3[dim - 1] = np.log(gen_test_error_Q3(data_set, data_set_y, test_set_x_Q3, test_set_y_Q3, dim))
plt.figure(figsize = (8, 6))
plt.plot(k_test, ln_MSE_test_Q3, label = 'Log of training error for test data of new basis')
plt.xlabel('dimension k', fontsize = 17)
plt.ylabel('Log(MSE)', fontsize = 17)
plt.xticks(k_test)
plt.legend()
plt.show()

##### Q3(d) #####
MSE_train_Q3 = []
MSE_test_Q3 = []
for dim in k_Q3:
    MSE_sum_train = 0
    MSE_sum_test = 0
    for i in range(100):

        x_train = np.random.uniform(0, 1, 30)
        x_test = np.random.uniform(0, 1, 1000)

        y_train = gen_dataset(x_train, 0.07, 30)
        y_test = gen_dataset(x_test, 0.07, 1000)

        MSE_sum_train += get_MSE_Q3(x_train, y_train, dim)
        MSE_sum_test += gen_test_error_Q3(x_train, y_train, test_set_x, y_test, dim)


    MSE_train_Q3.append(np.log(MSE_sum_train / 100))
    MSE_test_Q3.append(np.log(MSE_sum_test / 100))

plt.figure(figsize = (8, 6))
plt.plot(k_Q3, MSE_train_Q3, label = 'log avg of training error of new basis for 100 times ')
plt.plot(k_Q3, MSE_test_Q3, label = 'log avg of testing error of new basis for 100 times')
plt.xlabel('dimension k', fontsize = 17)
plt.ylabel('Log(avg MSE)', fontsize = 17)
plt.xticks(k_Q3)
plt.legend()
plt.show()

##### Q4 functions #####
Boston = pd.read_csv('Boston-filtered.csv')

def sample_dataset(data, sample_ratio):
    data = np.array(data)
    shuffle_index = np.random.permutation(len(data))
    sample_index = int(sample_ratio * len(data)) + 1
    train_data, test_data = data[shuffle_index[sample_index: ]], data[shuffle_index[: sample_index]]
    return train_data, test_data

def MSE_Q4(y, y_pre):
    MSE = []
    for i in range(len(y)):
        sse = (y[i] - y_pre[i])**2
        MSE.append(sse)
    return np.sum(MSE) / len(y)

def Naive_Regression(data, sample_ratio, run):
    MSE_train = []
    MSE_test = []

    for i in range(run):
        train_data, test_data = sample_dataset(data, sample_ratio)
        one_train = np.mat(np.ones(len(train_data))).T
        one_test = np.mat(np.ones(len(test_data))).T


        train_data_y = train_data[:,12].T
        test_data_y = test_data[:,12].T

        w = np.matmul(np.linalg.pinv(one_train), train_data_y)

        MSE_train_run = MSE_Q4(train_data_y, np.matmul(one_train, w))
        MSE_test_run = MSE_Q4(test_data_y, np.matmul(one_test, w))

        MSE_train.append(MSE_train_run)
        MSE_test.append(MSE_test_run)


    avg_MSE_train = np.mean(MSE_train)
    avg_MSE_test = np.mean(MSE_test)
    

    std_MSE_train = np.std(MSE_train)
    std_MSE_test = np.std(MSE_test)
    return avg_MSE_train, avg_MSE_test, std_MSE_train, std_MSE_test

def Linear_Regression_single(data, sample_ratio, run, index):
    
    MSE_train = []
    MSE_test = []

    for i in range(run):
        train_data, test_data = sample_dataset(data, sample_ratio)
        train_data_y = train_data[:,12]
        test_data_y = test_data[:,12]

        train_data_att = np.mat(train_data[:,index])
        test_data_att = np.mat(test_data[:,index])

        one_train = np.mat(np.ones(len(train_data)))
        one_test = np.mat(np.ones(len(test_data)))

        train_data_att_bias = np.concatenate((train_data_att, one_train), 0)
        test_data_att_bias = np.concatenate((test_data_att, one_test), 0)

        w = np.matmul(np.linalg.pinv(train_data_att_bias.T), train_data_y)

        MSE_train_att = MSE_Q4(train_data_y, np.matmul(train_data_att_bias.T, w.T))
        MSE_test_att = MSE_Q4(test_data_y, np.matmul(test_data_att_bias.T, w.T))

        MSE_train.append(MSE_train_att)
        MSE_test.append(MSE_test_att)
    
    avg_MSE_train = np.mean(MSE_train)
    avg_MSE_test = np.mean(MSE_test)


    std_MSE_train = np.std(MSE_train)
    std_MSE_test = np.std(MSE_test)
    return avg_MSE_train, avg_MSE_test, std_MSE_train, std_MSE_test

def Linear_Regresison_all(data, sample_ratio, run):
    MSE_train = []
    MSE_test = []

    for i in range(run):
        train_data, test_data = sample_dataset(data, sample_ratio)
        train_data_y = train_data[:,12]
        test_data_y = test_data[:,12]

        train_data_all = train_data[:, :12]
        test_data_all = test_data[:, :12]

        one_train = np.mat(np.ones(len(train_data))).T
        one_test = np.mat(np.ones(len(test_data))).T

        train_data_all_att = np.concatenate((train_data_all, one_train), 1)
        test_data_all_att = np.concatenate((test_data_all, one_test), 1)

        w = np.matmul(np.linalg.pinv(train_data_all_att), train_data_y)

        mse_train = MSE_Q4(np.matmul(train_data_all_att, w.T), train_data_y)
        mse_test = MSE_Q4(np.matmul(test_data_all_att, w.T), test_data_y)

        MSE_train.append(mse_train)
        MSE_test.append(mse_test)


    avg_MSE_train = np.mean(MSE_train)
    avg_MSE_test = np.mean(MSE_test) 

    std_MSE_train = np.std(MSE_train)
    std_MSE_test = np.std(MSE_test)

    return avg_MSE_train, avg_MSE_test, std_MSE_train, std_MSE_test


##### Q4 (a) #####
Boston_data = pd.read_csv('Boston-filtered.csv')
run = 20
test_ratio = 1/3
avg_MSE_train, avg_MSE_test, std_MSE_train, std_MSE_test= Naive_Regression(Boston_data, test_ratio, run)

print('MSE for training data over 20 runs is' + str(avg_MSE_train))
print('MSE for testing data over 20 runs is' + str(avg_MSE_test))


##### Q4 (c) #####
for i in range(12):
    avg_MSE_train, avg_MSE_test, std_MSE_train, std_MSE_test = Linear_Regression_single(Boston_data, test_ratio,run, i)

    print('Attribute' + str(i+1))
    print('MSE for training data over 20 runs is' + str(avg_MSE_train))
    print('MSE for testing data over 20 runs is' + str(avg_MSE_test))
    print('')

##### Q4 (d) #####
avg_MSE_train, avg_MSE_test, std_MSE_train, std_MSE_test = Linear_Regresison_all(Boston_data, test_ratio, run)

print('MSE for training data over 20 runs is' + str(avg_MSE_train))
print('MSE for testing data over 20 runs is' + str(avg_MSE_test))


##### Q5 functions #####

def Gaussian_kernel(xi, xj, sig):
    xi_row = xi.shape[0]
    xj_row = xj.shape[0]

    xi_sum = np.sum(xi**2, axis = 1)
    xj_sum = np.sum(xj**2, axis = 1)

    xi_sq = xi_sum.reshape((xi_row,1))
    xj_sq = xj_sum.reshape((1,xj_row))
    
    norm = xi_sq + xj_sq - 2* np.dot(xi, xj.T)
    
    result = np.exp(-norm / (2 * sig**2))
    return result


def gen_K_fold_data(data, K, i):
    x_data = data[:, :12]
    y_data = data[ :,12]
    single_fold = len(data) // K

    fold_index_begin = i * single_fold
    fold_index_end = (i + 1) * single_fold

    x_valid = x_data[fold_index_begin: fold_index_end]
    y_valid = y_data[fold_index_begin: fold_index_end]

    x_remain = np.concatenate((x_data[0:fold_index_begin], x_data[fold_index_end :]), 0)
    y_remain = np.concatenate((y_data[0:fold_index_begin], y_data[fold_index_end :]), 0)

    return x_valid, y_valid, x_remain, y_remain

def gen_err_matrix(gam, sig, K, data):
    err_matrix = np.zeros((len(gam), len(sig)))
    for g in range(len(gam)):
        for s in range(len(sig)):
            MSE = 0
            for i in range(K):
                x_valid, y_valid, x_train, y_train = gen_K_fold_data(data, K, i)
                K_train = Gaussian_kernel(x_train, x_train, sig[s])
                w = np.dot(np.linalg.inv(K_train + gam[g] * (K_train.shape[0]) * np.eye((K_train.shape[0]))), y_train)

                K_valid = Gaussian_kernel(x_train, x_valid, sig[s])

                err = np.matmul(w.T, K_valid) - y_valid
                MSE += np.matmul(err, err.T) / len(x_valid)
            err_matrix[g][s] = MSE/K
            
    
    return err_matrix

gamma = []
sigma = []
gamma_index = []
sigma_index = []
best_index = []
for i in range(-40, -25):
    gamma.append(2**i)
    gamma_index.append(i)


for i in np.arange(7, 13.5, 0.5):
    sigma.append(2**i)
    sigma_index.append(i)
            
gamma = np.array(gamma)
sigma = np.array(sigma)


##### Q5 (a) #####
Boston_data = pd.read_csv('Boston-filtered.csv')
sample_ratio = 1/3
train_data, test_data = sample_dataset(Boston_data, sample_ratio)

x_train = train_data[:, :12]
y_train = train_data[:, 12]

x_test = test_data[:, :12]
y_test = test_data[:, 12]

K = 5

##### Q5 b and c #####
error_matrix = gen_err_matrix(gamma, sigma, K, train_data)
for g in range(len(gamma)):
    for s in range(len(sigma)):
        if error_matrix[g][s] == np.min(error_matrix):

            best_index.append(g)
            best_index.append(s)

            print('The best gamma is 2^' + str(gamma_index[g]))
            print('The best sigma is 2^' + str(sigma_index[s]))

Kernel_train = Gaussian_kernel(x_train, x_train, sigma[best_index[1]])
Kernel_test = Gaussian_kernel(x_train, x_test, sigma[best_index[1]])

w = np.dot(np.linalg.inv(Kernel_train + gamma[best_index[0]] * (Kernel_train.shape[0]) * np.eye((Kernel_train.shape[0]))), y_train)

MSE_train = np.mean((np.dot(Kernel_train.T, w) - y_train)**2)
MSE_test = np.mean((np.dot(Kernel_test.T, w) - y_test)**2)


print ('The best training MSE = ' + str(MSE_train))
print ('The best test MSE = ' + str(MSE_test))



x, y = np.meshgrid(sigma_index, gamma_index)
fig = plt.figure(figsize = (20, 15))
ax = plt.axes(projection = '3d')
ax.set_xlabel('Gamma power i', fontsize = 17)
ax.set_ylabel('Sigma power j', fontsize = 17)
ax.set_zlabel('Average MSE', fontsize = 17)
surface = ax.plot_surface(x, y, error_matrix, cmap = 'rainbow')
fig.colorbar(surface)
plt.show()

##### Q5 d, repeat 4a, c, d#####
##### Q5 a #####
Boston_data = pd.read_csv('Boston-filtered.csv')
test_ratio = 1/3
run = 20

avg_MSE_train, avg_MSE_test, std_MSE_train, std_MSE_test= Naive_Regression(Boston_data, test_ratio, run)

print('Naive Regression')
print('MSE for training data over 20 runs is' + str(avg_MSE_train) + '±' + str(std_MSE_train))
print('MSE for testing data over 20 runs is' + str(avg_MSE_test) + '±' + str(std_MSE_test))


##### Q5 c #####
print('')
print('Linear Regression for single attribute')
for i in range(12):
    avg_MSE_train, avg_MSE_test, std_MSE_train, std_MSE_test = Linear_Regression_single(Boston_data, test_ratio,run, i)

    print('Attribute' + str(i+1))
    print('MSE for training data over 20 runs is' + str(avg_MSE_train) + '±' + str(std_MSE_train))
    print('MSE for testing data over 20 runs is' + str(avg_MSE_test) + '±' + str(std_MSE_test))
    print('')


##### Q5 d #####
print('')
print('Linear Regression for all attributes')
avg_MSE_train, avg_MSE_test, std_MSE_train, std_MSE_test = Linear_Regresison_all(Boston_data, test_ratio, run)

print('MSE for training data over 20 runs is' + str(avg_MSE_train) + '±' + str(std_MSE_train))
print('MSE for testing data over 20 runs is' + str(avg_MSE_test) + '±' + str(std_MSE_test))

##### repeat 5a and 5c #####


test_ratio = 1/3

train_mse = []
test_mse = []

for i in range(20):
    train_data, test_data = sample_dataset(Boston_data, 1/3)
    err_matrix = gen_err_matrix(gamma, sigma, K, train_data)

    for i in range(err_matrix.shape[0]):
        for j in range(err_matrix.shape[1]):
            if err_matrix[i,j] == np.min(err_matrix):

                best_gamma_index = gamma_index[i]
                best_sigma_index = sigma_index[j]

    print('The best gamma is 2^' + str(best_gamma_index))
    print('The best sigma is 2^' + str(best_sigma_index))
    print ('')

    best_gamma = 2 ** best_gamma_index
    best_sigma = 2** best_sigma_index

    x_train = train_data[:, :12]
    y_train = train_data[:, 12]
    x_test = test_data[:, :12]
    y_test = test_data[:, 12]


    k_train = Gaussian_kernel(x_train, x_train, best_sigma)
    k_test = Gaussian_kernel(x_train, x_test, best_sigma)

    w = np.matmul(np.linalg.pinv(k_train + best_gamma * len(k_train) * np.eye(len(k_train))), y_train)
    err_train = np.matmul(w.T, k_train) - y_train

    train_MSE = np.matmul(err_train, err_train.T)/ len(x_train)
    train_mse.append(train_MSE)


    err_test = np.matmul(w.T,  k_test) - y_test
    test_MSE = np.matmul(err_test, err_test.T) / len(x_test)
    test_mse.append(test_MSE)

avg_mse_train = np.mean(train_mse)
avg_mse_test = np.mean(test_mse)

std_mse_train = np.std(train_mse)
std_mse_test = np.std(test_mse)

print('Kernelized ridge regression')
print('MSE train = ' + str(avg_mse_train) + '±' + str(std_mse_train))
print('MSE test = ' + str(avg_mse_test) + '±' + str(std_mse_test))