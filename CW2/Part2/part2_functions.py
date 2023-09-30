import numpy as np




def random_sample(data, l):
    '''
    implement the wo replacement sample
    '''

    label_list = [1, 3]
    choose_list_2_labels = np.array([])
    
    for label in label_list:

        index_label = np.where(data[:, 0] == label)[0]
        #select l indexes randomly

        choose_index = np.random.choice(index_label, l, replace = False)

        choose_list_2_labels = np.concatenate((choose_list_2_labels, choose_index), axis = 0)
    
    return choose_list_2_labels

def get_adjacency_weight_matrix(data):
    # use 3-NN to generate a mxm adjacency matrix
    ad_matrix = np.zeros([data.shape[0], data.shape[0]])

    x = data[:,1:]
    y = data[:,0]

    # find 3 nearest neighbors for each record
    for i in range(data.shape[0]):
        x_current = x[i]
        # get a vector of distance
        distance = np.linalg.norm(x-x_current,ord=2,axis=1)
        # get the corresponding index of the ranked data, and exclude the x_current itself as it ranked the first
        corresponding_index = np.argsort(distance)[1:4]

        for neighbor in corresponding_index:
            ad_matrix[i][neighbor] = 1
            ad_matrix[neighbor][i] = 1

    return ad_matrix

def get_degree_matrix(ad_matrix):
    degree_matrix = np.zeros(ad_matrix.shape)
    for i in range(ad_matrix.shape[0]):
        num_non_zero = 0
        for num in ad_matrix[i]:
            if num:
                num_non_zero += 1
        degree_matrix[i][i] = num_non_zero
    return degree_matrix

def Laplacian_Interpolation(data, index_list):
    '''
    Implement the Laplacian Interpolation, where the index_list is the labeled ones,
    in this case, the consensus algorithm is conducted.
    '''

    y_data = data[:, 0].astype(int)
    x_data = data[:, 1:]

    y_pre_data = np.zeros(len(y_data))

    for index in index_list.astype(int):
        if y_data[index] == 3:
            y_pre_data[index] = 1
        
        if y_data[index] == 1:
            y_pre_data[index] = -1
    
    free_index = np.where(y_pre_data == 0)[0]

    weight_matrix = get_adjacency_weight_matrix(data)

    iter_n = data.shape[0] * 100
    # optimise the label
    for iter in range(iter_n):

        random_index = np.random.choice(free_index)

        y_pre_data[random_index] = np.dot(y_pre_data, weight_matrix[random_index]) / np.sum(weight_matrix[random_index])

    for i in range(len(y_pre_data)):
        
        if y_pre_data[i] > 0:
            y_pre_data[i] = 3

        if y_pre_data[i] <= 0:
            y_pre_data[i] = 1

    #test
    errors = np.sum(y_data != y_pre_data)
    return errors/(len(data) - len(index_list))



def LaplacianKernelInterpolation_cjc(data, label_list):
    # randomly generate labelled set
    y = data[:,0]
    

    label_size = len(label_list)

    ad_matrix = get_adjacency_weight_matrix(data)
    laplacian = get_degree_matrix(ad_matrix) - ad_matrix

    # generate kernel matrix
    kernel_matrix = np.empty((label_size, label_size))
    pseudoinv_laplacian = np.linalg.pinv(laplacian)
    for i in range(label_size):
        for j in range(label_size):
            kernel_matrix[i][j] = pseudoinv_laplacian[label_list[i]][label_list[j]]

    # generate labelled vector
    y_l = np.empty(label_size)
    for i in range(y_l.size):
        y_l[i] = y[label_list[i]]

    # generate alpha vector
    alpha_vector = np.linalg.pinv(kernel_matrix) @ y_l

    # generate coordinate matrix
    coordinate_matrix = np.zeros([label_size,laplacian.shape[1]])
    for i in range(coordinate_matrix.shape[0]):
        for j in range(coordinate_matrix.shape[1]):
            coordinate_matrix[i][j] = 0 if laplacian[label_list[i]][j] == 0 else 1

    v = np.zeros(laplacian.shape[0])
    for i in range(len(alpha_vector)):
        alpha = alpha_vector[i]
        e_i = coordinate_matrix[i,:]
        v += alpha * e_i.T @ pseudoinv_laplacian

    # generate discrete prediction vector
    prediction_vector = [1 if v[i] < 0 else 3 for i in range(v.size)]

    # calculate empirical generalisation error
    error = 0
    #labelledset_index_list = [i[0] for i in labelledset]
    for i in range(len(y)):
        if not i in label_list:
            predicted = prediction_vector[i]
            actual = y[i]
            if not (predicted == actual):
                error += 1

    return error / (len(prediction_vector) - len(label_list))

