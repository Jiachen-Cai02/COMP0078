from part1_functions import polynomial_kernel, data_split, OvR_poly_kernel, test_OvR_poly
import numpy as np

data = np.loadtxt('zipcombo.dat')
y = data[:, 0].astype(int)
x = data[:, 1:]

x_train, x_test, y_train, y_test = data_split(x, y)
alpha = np.zeros((10, x_train.shape[0]))

Q1 = []

for i in range(1, 8):
    train_error = []
    test_error = []
    for run in range(20):
        x_train, x_test, y_train, y_test = data_split(x, y)
        list_i, alpha_i = OvR_poly_kernel(x_train, y_train, i, alpha, 30)
        test_err = test_OvR_poly(x_train, x_test, y_test, alpha_i, i)

        train_error.append(list_i[-1])
        test_error.append(test_err)

    Q1.append([np.mean(train_error) * 100, np.std(train_error) * 100, np.mean(test_error) * 100, np.std(test_error) * 100])

        
print(Q1)




