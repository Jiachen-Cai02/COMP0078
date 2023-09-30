from part1_functions import Gaussian_kernel, data_split, OvR_Gaussian_kernel, test_OvR_Gaussian
import numpy as np

data = np.loadtxt('zipcombo.dat')
y = data[:, 0].astype(int)
x = data[:, 1:]

x_train, x_test, y_train, y_test = data_split(x, y)
alpha = np.zeros((10, x_train.shape[0]))

Q1 = []
order = np.zeros(len(y_test))
for i in range(1, 8):
    train_error = []
    test_error = []
    for run in range(20):
        list_i, alpha_i = OvR_Gaussian_kernel(x_train, y_train, 3 ** -i, alpha, 30)
        test_err = test_OvR_Gaussian(x_train, x_test, y_test, alpha_i, 3 ** -i)

        train_error.append(list_i[-1])
        test_error.append(test_err)

    Q1.append([np.mean(train_error) * 100, np.std(train_error) * 100, 
    np.mean(test_error) * 100, np.std(test_error) * 100])

        
print(Q1)