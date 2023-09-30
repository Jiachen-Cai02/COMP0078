import numpy as np
import matplotlib.pyplot as plt
from part1_functions import data_split, OvR_poly_kernel, test_OvR_poly, cross_validation, confusion_matrix

data = np.loadtxt('zipcombo.dat')
y = data[:, 0].astype(int)
x = data[:, 1:]
e = 8

x_train, x_test, y_train, y_test = data_split(x, y)
alpha = np.zeros((10, x_train.shape[0]))


Q2 = []
matrix = np.zeros((20, 10, 10))
for run in range(20):
    x_train, x_test, y_train, y_test = data_split(x, y)
    k_fold_data = cross_validation(x_train, y_train, 5)
    error_run = []

    for d in range(1, 8):
        
        error = []

        for n in range(5):
            x_train_n, x_test_n, y_train_n, y_test_n = k_fold_data[n]
            alpha = np.zeros((10, x_train_n.shape[0]))

            list_n, alpha_n = OvR_poly_kernel(x_train_n, y_train_n, d, alpha, 30)

            test_err = test_OvR_poly(x_train_n, x_test_n, y_test_n, alpha_n, d)

            error.append(test_err)
    
        error_run.append(np.mean(error))
    
    best_d = np.argmin(error_run) + 1

    alpha = np.zeros((10, x_train.shape[0]))
    min_list, best_alpha = OvR_poly_kernel(x_train, y_train, best_d, alpha, 30)
    test_error, matrix[run] = test_OvR_poly(x_train, x_test, y_test, best_alpha, best_d, record = True)
    #matrix[n] = confusion_matrix(x_train, x_test, y_test, best_alpha, best_d, matrix[n])

    Q2.append([best_d, test_error])

print(Q2)
print(np.mean(matrix, axis = 0))
print(np.std(matrix, axis = 0))

'''
plt.figure(figsize = (20, 8))

for n in list(enumerate(hardest_five)):
    plt.subplot(2, 5, n[0] + 1)
    plt.imshow(x[n[1], :].reshape(16,16), cmap = "gray")
    plt.title('label = %.0f' % int(y[n[1]]), fontsize = 20)
    plt.axis('off')
'''
