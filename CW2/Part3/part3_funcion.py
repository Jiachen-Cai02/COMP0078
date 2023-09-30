import numpy as np


def generate_data(m, n, method):
    '''
    Implement the method for generate data, for winnow is {0, 1}^n
    '''
    if method == Winnow_method:

        x_data = np.random.choice([0, 1], (m, n))
        y_data = x_data[:, 0]
    
    else:

        x_data = np.random.choice([1, -1], (m, n))
        y_data = x_data[:, 0]
    
    return x_data, y_data

def Winnow_method(x, y, x_test):

    w = np.ones(x.shape[1])
    n = x.shape[1]

    #train
    for t in range(len(x)):

        y_pre = np.where(w @ x[t] < n, 0, 1)

        if y_pre != y[t]:

            w*= np.float_power(2, ((y[t] - y_pre) * x[t]))
        else:
            continue
    
    y_test = np.where(x_test @ w < n, 0, 1)

    return y_test

def perceptron(x, y, x_test):
    '''
    Implement the perceptron method on the lecture notes
    '''

    w = np.zeros(x.shape[1])
    M = 0

    for t in range(len(x)):
        y_pre = np.sign(np.dot(w, x[t]))

        if y_pre * y[t] <= 0:
            w += y[t] * x[t]
            M += 1
    
    y_t = np.sign(np.dot(x_test, w))

    return y_t
    
def onenn (x, y, x_test):
    '''
    Implement 1nn method which is the same as in the previous section
    '''
    y_t = np.zeros(x_test)

    for i in range(len(x_test)):

        x_point = x_test[i]
        distance_list = np.sum((x - x_point) **2, axis = 1)
        y_index = np.argmin(distance_list)

        y_t[i] = y[y_index]
    
    return y_t


def least_square(x, y, x_test):
    '''
    Implement the linear regression method in the question
    '''

    w = np.linalg.pinv(x) @ y
    y_test = np.sign(x_test @ w)

    return y_test

def generalisation_error(m, n, m_test, I, method):

    error_list = []

    for i in range(I):
        x_train, y_train = generate_data(m, n, method)
        x_test, y_test = generate_data(m_test, n, method)

        y_pred = method(x_train, y_train, x_test)

        error = np.sum(y_pred != y_test)

        error_list.append(error/m_test)

    avg_error = np.mean(error_list)

    return avg_error

def sample_complexity(N, m_test, I, method):

    '''
    Implement the estimated sample complexity method, where I is the loop which is the same as last function
    However, N here is a list of size that to derive the minimize size for sample complexity
    '''

    complexity = np.zeros((I, N-1))
    
    for i in range(I):

        #start to loop
        for n in range(1, N):
            m = 1
            avg_generalisation_error = generalisation_error(m, n, m_test, I, method)

            while avg_generalisation_error > 0.1:
                #stopping creteria
                m += 1
                avg_generalisation_error = generalisation_error(m, n, m_test, I, method)
            
            complexity[i][n-1] = m

    avg_complexity = np.mean(complexity, axis = 0)
    std_complexity = np.std(complexity, axis = 0)

    return avg_complexity, std_complexity



