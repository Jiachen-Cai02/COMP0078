import numpy as np
import matplotlib.pyplot as plt
from part1_functions import data_split, OvO_poly_kernel, test_OvO_poly, cross_validation

data = np.loadtxt('zipcombo.dat')
y = data[:, 0].astype(int)
x = data[:, 1:]
e = 8

x_train, x_test, y_train, y_test = data_split(x, y)
alpha = np.zeros((10, x_train.shape[0]))


Q2 = []
matrix = np.zeros((20, 10, 10))
order = np.zeros(len(y_test))
for run in range(20):
    x_train, x_test, y_train, y_test = data_split(x, y)
    k_fold_data = cross_validation(x_train, y_train, 5)
    error_run = []

    for d in range(1, 8):
        
        error = []

        for n in range(5):
            x_train_n, x_test_n, y_train_n, y_test_n = k_fold_data[n]
            alpha = np.zeros((10, x_train_n.shape[0]))

            list_n, alpha_n = OvO_poly_kernel(x_train_n, y_train_n, d, 30)

            test_err = test_OvO_poly(x_train_n, x_test_n, y_test_n, alpha_n, d)

            error.append(test_err)
    
        error_run.append(np.mean(error))
    
    best_d = np.argmin(error_run) + 1

    alpha = np.zeros((10, x_train.shape[0]))
    min_list, best_alpha = OvO_poly_kernel(x_train, y_train, best_d, 30)
    test_error = test_OvO_poly(x_train, x_test, y_test, best_alpha, best_d)

    Q2.append([best_d, test_error])

print(Q2)
