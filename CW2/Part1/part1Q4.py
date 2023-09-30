import numpy as np
from part1_functions import *

data = np.loadtxt('zipcombo.dat')
y = data[:, 0].astype(int)
x = data[:, 1:]
e = 8

x_train, x_test, y_train, y_test = data_split(x, y)
alpha = np.zeros((10, x_train.shape[0]))

Q4_data = np.zeros(len(y))

for run in range(50):
    y_pred = np.zeros(len(y))

    x_train, x_test, y_train, y_test = data_split(x, y)
    d_list, alpha_best = OvR_poly_kernel(x_train, y_train, 4, alpha, 30)


    kernel_matrix_for_all_data = polynomial_kernel(x, x_train, 4)
    for i in range(len(kernel_matrix_for_all_data)):
        max_val = -float('inf')
        best_label = -1
        for n in range(10):
            if y[i] == n:
                y_for_now = 1
            else:
                y_for_now = -1
            
            pred_val = np.dot(alpha[n], kernel_matrix_for_all_data[i])

            if pred_val > max_val:
                max_val = pred_val
                best_label = n
        
        y_pred[i] = best_label
    
    Q4_data[np.argwhere(y_pred != y)] += 1

hardest = np.argsort(Q4_data)[-5 :]

print(hardest)

plt.figure(figsize=(20, 8))
for i, j in enumerate(hardest):
    plt.subplot(2, 5, i+1)
    plt.imshow(x[j, :].reshape(16, 16), cmap = "gray")
    plt.title('label = %.0f' % int(y[j]), fontsize = 20)
    plt.axis('off')

plt.show()

