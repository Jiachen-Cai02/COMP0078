import numpy as np
import matplotlib.pyplot as plt


def gen_data(num):

    x = np.random.uniform (size = 2 * num).reshape(num,2)

    y = np.random.randint(0,2, size = num)

    return x, y


def KNN_L2 (x_data, y_data, t_data, K):
    pre_list = []

    #For this question, the L2 KNN would be inducted
    for t in t_data:

        distance = np.linalg.norm(x_data - t, ord = 2, axis = 1)
    
        sort_y_data = y_data[np.argsort(distance)]

        y = np.mean(sort_y_data[:K])

        if y > 0.5:    
            pre = 1
        elif y < 0.5:
            pre = 0
        else:
            pre = np.random.randint(0,2)
        
        pre_list.append(pre)
    return np.array(pre_list)



def KNN_plt (x_data, y_data, K):

    x_background, y_backgrond = np.meshgrid(np.arange(0, 1.1, 0.01), np.arange(0, 1.1, 0.01))
    
    row, col = x_background.shape

    x_y_background = np.c_[x_background.ravel(), y_backgrond.ravel()]

    pre = KNN_L2(x_data, y_data, x_y_background, K).reshape(row, col)

    plt.figure(figsize= (8, 6))

    plt.contourf(x_background, y_backgrond, pre, cmap = 'gist_earth')
    plt.scatter(x_data[:, 0], x_data[:, 1], c = y, cmap = 'gist_earth')
    plt.show()


x, y = gen_data(100)
KNN_plt(x, y, 3)

def gen_ph (x_data, y_data, N, K):

    y = []
    x = np.random.uniform(size = N * 2).reshape(N,2)

    for n in range(N):
        con = np.random.choice([0,1], p = [0.2, 0.8])

        if con == 0:
            #When the con is heads
            y.append(np.random.randint(0,2))

        elif con == 1:
            #When the con is tails
            y.append(KNN_L2(x_data, y_data, x[n, :].reshape(-1,2), K).item())

    return x, np.array(y)

def Protocol_A (train_n, test_n, K):
    error_list = []

    for k in range(1, 50):
        error = 0
        for n in range(100):
            err = 0
            x, y = gen_data(100)

            x_train, y_train = gen_ph(x, y, train_n, K)
            x_test, y_test = gen_ph(x, y, test_n, K)

            pre = KNN_L2(x_train, y_train, x_test, k)
            
            for n in range(len(pre)):
                if pre[n] != y_test[n]:
                    err += 1
            error += err/len(pre)

        error_list.append(error / 100)

    return error_list


error = Protocol_A(4000, 1000, 3)

plt.figure(figsize = (8, 5))

plt.plot(range(1, 50), error)
plt.title('generalised error against k', fontsize = 17)
plt.xlabel('k-value', fontsize = 14)
plt.ylabel('average generalised error', fontsize = 14)
plt.show()

def protocol_B(K):

    Best_K = []
    for m in [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]:

        optimal_K = 0
        for i in range(100):
            k_error = []
            for k in range(1, 50):
                err = 0
                x, y = gen_data(100)

                x_train, y_train = gen_ph(x, y, m, K)
                x_test, y_test = gen_ph(x, y, 1000, K)

                pre = KNN_L2(x_train, y_train, x_test, k)

                for n in range(len(pre)):
                    if pre[n] != y_test[n]:
                        err += 1
                k_error.append(err/len(pre))
            optimal_K += (np.argmin(k_error) +1)
        Best_K.append(optimal_K/100)
    
    return Best_K
    
optimal_k  = protocol_B(3)

plt.figure(figsize = (8, 5))
plt.plot((100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000), optimal_k)
plt.title('average optimal k against training size', fontsize = 17)
plt.ylabel('average optimal k-value', fontsize = 14)
plt.xlabel('training size', fontsize = 14)
plt.show()