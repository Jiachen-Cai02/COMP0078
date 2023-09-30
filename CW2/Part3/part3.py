import numpy as np
import matplotlib.pyplot as plt
from part3_funcion import *

N = 101
m_test = 5000
I = 10

avg_perceptron, std_proceptron = sample_complexity(N, m_test, I, perceptron)

plt.figure(figsize = (8, 5))
plt.errorbar(range(1, N), avg_perceptron, yerr = std_proceptron, ecolor = 'salmon', capsize = 5)
plt.xlabel("n")
plt.ylabel("m")
plt.savefig('perceptron', dpi = 500)
plt.show()

avg_Winnow, std_Winnow = sample_complexity(N, m_test, I, Winnow_method)

plt.figure(figsize = (8, 5))
plt.errorbar(range(1, N), avg_Winnow, yerr = std_Winnow, ecolor = 'salmon', capsize = 5)
plt.xlabel("n")
plt.ylabel("m")
plt.savefig('Winnow', dpi = 500)
plt.show()

avg_1nn, std_1nn = sample_complexity(N, m_test, I, onenn)

plt.figure(figsize = (8, 5))
plt.errorbar(range(1, N), avg_1nn, yerr = std_1nn, ecolor = 'salmon', capsize = 5)
plt.xlabel("n")
plt.ylabel("m")
plt.savefig('One NN', dpi = 500)
plt.show()

avg_ls, std_ls = sample_complexity(N, m_test, I, least_square)

plt.figure(figsize = (8, 5))
plt.errorbar(range(1, N), avg_ls, yerr = std_ls, ecolor = 'salmon', capsize = 5)
plt.xlabel("n")
plt.ylabel("m")
plt.savefig('Least Square', dpi = 500)
plt.show()