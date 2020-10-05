import numpy as np
from mpmath import mp

mp.dps = 500


def compute_naive_p(test_statistic, n_a, n_b, sigma):

    z = test_statistic / (sigma * np.sqrt(1 / n_a + 1 / n_b))

    naive_p = mp.ncdf(z)

    return float(naive_p)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def construct_z(binary_vec, list_zk, list_results):
    z_interval = []

    for i in range(len(list_results)):
        if np.array_equal(binary_vec, list_results[i]):
            z_interval.append([list_zk[i], list_zk[i + 1] - 1e-10])

    new_z_interval = []

    # for each_interval in z_interval:
    #     if len(new_z_interval) == 0:
    #         new_z_interval.append(each_interval)
    #     else:
    #         sub = each_interval[0] - new_z_interval[-1][1]
    #         if abs(sub) < 0.01:
    #             new_z_interval[-1][1] = each_interval[1]
    #         else:
    #             new_z_interval.append(each_interval)
    #
    # z_interval = new_z_interval
    return z_interval


def create_X_para(X_test, d):
    X_3D = X_test[0]

    X_2D = []

    for element in X_3D:
        X_2D.append(list(element.flatten()))

    X_2D = np.array(X_2D)

    X_vec = (X_2D.flatten()).reshape((d * d, 1))

    X_test = []

    for i in range(d):
        X_test.append([])

        for j in range(d):
            index = i * d + j
            pT = np.zeros(d * d)
            pT[index] = 1
            pT = (pT.reshape((d*d, 1))).T

            X_test[i].append([[pT, 0]])

    return np.array([X_test]), X_vec


def create_X_pad(X_para, d, no_channel):
    X_para_pad = []

    X_para = X_para[0]

    for i in range(d + 2):
        X_para_pad.append([])

        for j in range(d + 2):
            X_para_pad[i].append([])
            for k in range(no_channel):
                if (i == 0) or (j == 0) or (i == (d + 1)) or (j == (d + 1)):
                    pT = np.zeros(d * d)
                    pT = (pT.reshape((d * d, 1))).T
                    X_para_pad[i][j].append([pT, 0])
                else:
                    X_para_pad[i][j].append(X_para[i-1][j-1][k])

    return np.array([X_para_pad])


def conv(X_test, X_para_pad, kernel):
    # X_test: d x d x channel
    output = []
    output_para = []

    _, d, _, no_channel = X_test.shape
    _, w, _, no_filter = kernel.shape

    X_test = X_test[0]
    X_para_pad = X_para_pad[0]

    for i in range(1, d + 1):
        output.append([])
        output_para.append([])

        for j in range(1, d + 1):
            output[i - 1].append([])
            output_para[i - 1].append([])

            for filter_idx in range(no_filter):
                sum = 0
                sum_para = np.array([(np.zeros(d * d).reshape((d * d, 1))).T, 0])

                for k in range(no_channel):
                    X_k = X_test[:, :, k]
                    X_k = np.pad(X_k, ((1, 1), (1, 1)), 'constant')

                    X_k_para = X_para_pad[:, :, k, :]

                    kernel_k = kernel[:, :, k, filter_idx]

                    sum = sum + \
                          X_k[i - 1, j - 1] * kernel_k[0, 0] + \
                          X_k[i - 1, j] * kernel_k[0, 1] + \
                          X_k[i - 1, j + 1] * kernel_k[0, 2] + \
                          X_k[i, j - 1] * kernel_k[1, 0] + \
                          X_k[i, j] * kernel_k[1, 1] + \
                          X_k[i, j + 1] * kernel_k[1, 2] + \
                          X_k[i + 1, j - 1] * kernel_k[2, 0] + \
                          X_k[i + 1, j] * kernel_k[2, 1] + \
                          X_k[i + 1, j + 1] * kernel_k[2, 2]

                    sum_para = sum_para + \
                            X_k_para[i - 1, j - 1] * kernel_k[0, 0] + \
                            X_k_para[i - 1, j] * kernel_k[0, 1] + \
                            X_k_para[i - 1, j + 1] * kernel_k[0, 2] + \
                            X_k_para[i, j - 1] * kernel_k[1, 0] + \
                            X_k_para[i, j] * kernel_k[1, 1] + \
                            X_k_para[i, j + 1] * kernel_k[1, 2] + \
                            X_k_para[i + 1, j - 1] * kernel_k[2, 0] + \
                            X_k_para[i + 1, j] * kernel_k[2, 1] + \
                            X_k_para[i + 1, j + 1] * kernel_k[2, 2]

                    # check_sum = np.dot(sum_para[0], X_vec)[0][0] + sum_para[1]
                    # print(np.around(sum - check_sum, 3))

                output[i - 1][j - 1].append(sum)
                output_para[i - 1][j - 1].append(sum_para)

    output = np.array([output])
    output_para = np.array([output_para])

    return output, output_para


def max_pooling_event(e1, e2, e3, e4):
    return [
        np.array(e2) - np.array(e1),
        np.array(e3) - np.array(e1),
        np.array(e4) - np.array(e1)
    ]


def max_pooling(input, input_para):

    list_ineq = []

    input = input[0]
    input_para = input_para[0]

    d, _, no_channel = input.shape

    output = []
    output_para = []

    for i in range(0, d, 2):
        output.append([])
        output_para.append([])

        for j in range(0, d, 2):
            output[-1].append([])
            output_para[-1].append([])

            for k in range(no_channel):
                list_local_event = None

                X_k = input[:, :, k]
                X_k_para = input_para[:, :, k, :]

                max_val = max(X_k[i, j], X_k[i, j+1], X_k[i+1, j], X_k[i+1, j+1])
                output[-1][-1].append(max_val)

                max_idx = np.argmax([X_k[i, j], X_k[i, j+1], X_k[i+1, j], X_k[i+1, j+1]])
                if max_idx == 0:
                    output_para[-1][-1].append(X_k_para[i, j])
                    list_local_event = max_pooling_event(X_k_para[i, j], X_k_para[i, j+1], X_k_para[i+1, j], X_k_para[i+1, j+1])
                elif max_idx == 1:
                    output_para[-1][-1].append(X_k_para[i, j + 1])
                    list_local_event = max_pooling_event(X_k_para[i, j + 1], X_k_para[i, j], X_k_para[i + 1, j], X_k_para[i + 1, j + 1])
                elif max_idx == 2:
                    output_para[-1][-1].append(X_k_para[i + 1, j])
                    list_local_event = max_pooling_event(X_k_para[i + 1, j], X_k_para[i, j], X_k_para[i, j + 1], X_k_para[i + 1, j + 1])
                else:
                    output_para[-1][-1].append(X_k_para[i + 1, j + 1])
                    list_local_event = max_pooling_event(X_k_para[i + 1, j + 1], X_k_para[i, j], X_k_para[i, j + 1], X_k_para[i + 1, j])

                for element in list_local_event:
                    list_ineq.append(element)

    output = np.array([output])
    output_para = np.array([output_para])

    return output, output_para, list_ineq


def up_sampling(input, input_para):
    input = input[0]
    input_para = input_para[0]

    d, _, no_channel = input.shape

    output = []
    output_para = []

    for i in range(d):
        output.append([])
        output_para.append([])

        for j in range(d):
            output[-1].append([])
            output_para[-1].append([])

            for k in range(no_channel):
                X_k = input[:, :, k]
                val = X_k[i, j]
                output[-1][-1].append(val)

                X_k_para = input_para[:, :, k, :]
                val_para = X_k_para[i, j]
                output_para[-1][-1].append(val_para)

            output[-1].append(output[-1][-1])
            output_para[-1].append(output_para[-1][-1])

        output.append(output[-1])
        output_para.append(output_para[-1])

    output = np.array([output])
    output_para = np.array([output_para])

    return output, output_para


def compute_u_v(x, eta, d):
    sq_norm = (np.linalg.norm(eta)) ** 2

    e1 = np.identity(d) - (np.dot(eta, eta.T)) / sq_norm
    u = np.dot(e1, x)

    v = eta / sq_norm

    return u, v


def construct_test_statistic(x, binary_vec, d):
    vector_1_S_a = np.zeros(d)
    vector_1_S_b = np.zeros(d)

    n_a = 0
    n_b = 0

    for i in range(d):
        if binary_vec[i] == 0:
            n_a = n_a + 1
            vector_1_S_a[i] = 1.0

        elif binary_vec[i] == 1:
            n_b = n_b + 1
            vector_1_S_b[i] = 1.0

    if (n_a == 0) or (n_b == 0):
        return None, None

    vector_1_S_a = np.reshape(vector_1_S_a, (vector_1_S_a.shape[0], 1))
    vector_1_S_b = np.reshape(vector_1_S_b, (vector_1_S_b.shape[0], 1))

    first_element = np.dot(vector_1_S_a.T, x)[0][0]
    second_element = np.dot(vector_1_S_b.T, x)[0][0]

    etaTx = first_element / n_a - second_element / n_b

    eta = vector_1_S_a / n_a - vector_1_S_b / n_b

    return eta, etaTx


def pivot_with_specified_interval(z_interval, eta, etaTx, cov, tn_mu):
    tn_sigma = np.sqrt(np.dot(np.dot(eta.T, cov), eta))[0][0]
    # print(tn_sigma)
    numerator = 0
    denominator = 0

    for each_interval in z_interval:
        al = each_interval[0]
        ar = each_interval[1]

        denominator = denominator + mp.ncdf((ar - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)

        if etaTx >= ar:
            numerator = numerator + mp.ncdf((ar - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)
        elif (etaTx >= al) and (etaTx < ar):
            numerator = numerator + mp.ncdf((etaTx - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)

    if denominator != 0:
        return float(numerator/denominator)
    else:
        return None