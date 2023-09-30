import numpy as np
import matplotlib.pyplot as plt
import itertools

def polynomial_kernel(p, q, d):
    '''
    the function of the polynomial kernel for perceptron
    '''
    return np.dot(p, q.T) ** d 

def Gaussian_kernel(p, q, c):
    '''
    the function of the gaussian kernel for perceptron
    '''
    p_norm = np.sum(p ** 2, axis = -1)
    q_norm = np.sum(q ** 2, axis = -1)

    kernel_matrix = np.exp(-c * (p_norm[:, None] + q_norm[None, :] - 2* (np.dot(p, q.T))))


    return kernel_matrix




def data_split(x_data, y_data, ratio = 0.8):
    '''
    generate random splited train data and test data
    '''

    random_index = list(range(len(x_data)))
    np.random.shuffle(random_index)

    split_index = int(ratio * len(x_data))

    x_train, x_test = x_data[random_index[: split_index]], x_data[random_index[split_index :]]
    y_train, y_test = y_data[random_index[: split_index]], y_data[random_index[split_index :]]

    return x_train, x_test, y_train, y_test


def cross_validation(x_data, y_data, n):
    '''
    generate n folds cross validation data
    '''

    split_index = int(len(x_data) / n)

    cross_validation_data = []

    for i in range(n):
        start = i * split_index
        end = (i + 1) * split_index
        
        x_train = np.concatenate((x_data[0 : start], x_data[end :]))
        x_test = x_data[start : end]

        y_train = np.concatenate((y_data[0 : start], y_data[end :]))
        y_test = y_data[start : end]

        cross_validation_data.append([x_train, x_test, y_train, y_test])

    return cross_validation_data


def OvR_poly_kernel(x_data, y_data, d, alpha, epoch):
    '''
    To implement the OvR, we treat the n class classification 
    as n binary classification question, the label of the chosen class
    is 1 and the rest is -1
    '''
    ker_matrix = polynomial_kernel(x_data, x_data, d)
    mistake_list = []
    for e in range(epoch):
        mistake = 0
        alpha_last_time = alpha
        for i in range(len(ker_matrix)):
            max_val = -float('inf')
            best_label = -1
            for n in range(10):
                if y_data[i] == n:
                    y_for_now = 1
                else:
                    y_for_now = -1
            
                pred_val = np.dot(alpha[n], ker_matrix[i])

                if np.sign(pred_val) != y_for_now:
                    if np.sign(pred_val) == 0:
                        alpha[n][i] = y_for_now
                    alpha[n][i] -= np.sign(pred_val)
                else:
                    if pred_val > max_val:
                        max_val = pred_val
                        best_label = n
        
            if best_label != y_data[i]:
                mistake += 1
        
        #save the train error for this epoch
        mistake_list.append(mistake/len(x_data))
        
        ## check converge
        if e > 1:
            if mistake_list[e] < mistake_list[e - 1]:
                continue
            else:
                #print('training converge when epoch equals to', e)
                return mistake_list, alpha_last_time
        
    return mistake_list, alpha
                
def test_OvR_poly(x_train, x_test, y_test, alpha, d, record = False):
    '''
    implement the same method for the training method to test
    '''

    mistake = 0
    ker_matrix = polynomial_kernel(x_test, x_train, d)
    con_matrix = np.zeros((10, 10))
    for i in range(len(ker_matrix)):
        max_val = -float('inf')
        best_label = -1

        for n in range(10):
            y_for_now = 0

            if y_test[i] != n:
                y_for_now = -1
            else:
                y_for_now = 1
            
            pred_val = np.dot(ker_matrix[i], alpha[n])

            if pred_val > max_val:
                max_val = pred_val
                best_label = n
        
        if y_test[i] != best_label:
            if record:
                con_matrix[y_test[i]][best_label] += 1
            mistake += 1

    con_matrix = np.nan_to_num(con_matrix.T / np.sum(con_matrix, axis = 1)).T
    if record:   
        return mistake/len(x_test), con_matrix
    else:
        return mistake/len(x_test)

def confusion_matrix(x_train, x_test, y_test, alpha, d, matrix):

    ker_matrix = polynomial_kernel(x_test, x_train, d)

    for i in range(len(ker_matrix)):
        max_val = -float('inf')
        best_label = -1

        for n in range(10):
            y_for_now = 0

            if y_test[i] != n:
                y_for_now = -1
            else:
                y_for_now = 1
            
            pred_val = np.dot(ker_matrix[i], alpha[n])

            if pred_val > max_val:
                max_val = pred_val
                best_label = n
        
        if best_label != y_test[i]:
            matrix[y_test[i]][best_label] += 1
    
    matrix = matrix / len(ker_matrix)
    return matrix


def OvR_Gaussian_kernel(x_data, y_data, d, alpha, epoch):
    '''
    To implement the OvR, we treat the n class classification 
    as n binary classification question, the label of the chosen class
    is 1 and the rest is -1
    '''
    ker_matrix = Gaussian_kernel(x_data, x_data, d)
    mistake_list = []
    for e in range(epoch):
        mistake = 0

        for i in range(len(ker_matrix)):
            max_val = -float('inf')
            best_label = -1
            for n in range(10):
                if y_data[i] == n:
                    y_for_now = 1
                else:
                    y_for_now = -1
            
                pred_val = np.dot(ker_matrix[i], alpha[n])

                if np.sign(pred_val) != y_for_now:
                    if np.sign(pred_val) == 0:
                        alpha[n][i] = y_for_now
                    alpha[n][i] -= np.sign(pred_val)
                else:
                    if pred_val > max_val:
                        max_val = pred_val
                        best_label = n
        
            if best_label != y_data[i]:
                mistake += 1

        #save the train error for this epoch
        mistake_list.append(mistake/len(x_data))

        if e > 0 and mistake_list[-1] - mistake_list[-2] < 0.01:
            break
        else:
            continue
        
    return mistake_list, alpha

def test_OvR_Gaussian(x_train, x_test, y_test, alpha, d):
    '''
    implement the same method for the training method to test
    '''

    mistake = 0
    ker_matrix = Gaussian_kernel(x_test, x_train, d)
    for i in range(len(ker_matrix)):
        max_val = -float('inf')

        for n in range(10):
            y_for_now = 0

            if y_test[i] != n:
                y_for_now = -1
            else:
                y_for_now = 1
            
            pred_val = np.dot(ker_matrix[i], alpha[n])

            if pred_val > max_val:
                max_val = pred_val
                best_label = n
        
        if best_label != y_test[i]:
            mistake += 1

    return mistake/len(x_test)


def vote_confidence(confidence):

    label_group = list(itertools.combinations(range(10), 2))
    vote_classes = np.zeros(10)
    vote = list(((np.sign(confidence) + 2) // 2).astype(int))

    for m, n in enumerate(vote):
        vote_classes[label_group[m][int(n)]] += 1

    return np.argmax(vote_classes)

def OvO_poly_kernel(x_data, y_data, d, epoch):
    '''
    To implement the OvO, generate 45 classifier for 10 classes and vote for
    the final class for data
    '''
    ker_matrix = polynomial_kernel(x_data, x_data, d)
    mistake_list = []
    label_group = list(itertools.combinations(range(10), 2))

    alpha = np.zeros((45, len(x_data)))

    for e in range(epoch):
        mistake = 0
        for i in range(len(x_data)):

            confidence = np.dot(alpha, ker_matrix[i])
            pre_label = vote_confidence(confidence)

            if pre_label != y_data[i]:
                mistake += 1
            
            #update the alpha
            for index, (label_1, label_2) in enumerate(label_group):

                if y_data[i] == label_1 and np.sign(confidence[index]) != -1:
                    alpha[index, i] -= 1
                
                if y_data[i] == label_2 and np.sign(confidence[index]) != 1:
                    alpha[index, i] += 1
            
        mistake_list.append(mistake/len(x_data))

            ## check converge
        if e > 1:
            if mistake_list[e] - mistake_list[e - 1] < 0.1:
                break
            else:
                continue
        
    return mistake_list, alpha

def test_OvO_poly(x_train, x_test, y_test, alpha, d):

    '''
    implement the same method for the training method to test
    '''

    ker_matrix = polynomial_kernel(x_train, x_test, d)


    y_pred = np.zeros(len(x_test))
    
    for i in range(len(y_pred)):

        y_pred[i] = vote_confidence(np.dot(alpha, ker_matrix[:, i]))

    y_pred = y_pred.astype(int)


    return np.sum(y_pred != y_test) / len(y_test)