import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import time

import gen_data
import util


def run():
    d = 8
    IMG_WIDTH = d
    IMG_HEIGHT = d
    IMG_CHANNELS = 1
    mu_1 = 0
    mu_2 = 2

    global_list_ineq = []

    X_test, Y_test = gen_data.generate(1, IMG_WIDTH, mu_1, mu_2)

    X_para, X_vec = util.create_X_para(X_test, d)

    X_para_pad = util.create_X_pad(X_para, d, IMG_CHANNELS)

    model = load_model('./model/test_' + str(d) + '.h5')
    # model.summary()

    weights = model.get_weights()

    kernel_1 = weights[0]
    bias_1 = weights[1]

    kernel_2 = weights[2]
    bias_2 = weights[3]

    out_conv_1, out_conv_1_para = util.conv(X_test, X_para_pad, kernel_1)

    _, d, _, no_channel = out_conv_1.shape

    out_conv_1 = out_conv_1 + bias_1

    for i in range(d):
        for j in range(d):
            for k in range(no_channel):
                out_conv_1_para[0][i][j][k][1] = out_conv_1_para[0][i][j][k][1] + bias_1[k]

    out_max_pooling, out_max_pooling_para, max_pooling_event = util.max_pooling(out_conv_1, out_conv_1_para)

    for element in max_pooling_event:
        global_list_ineq.append(element)

    out_up_sampling, out_up_sampling_para = util.up_sampling(out_max_pooling, out_max_pooling_para)

    _, d, _, no_channel = out_up_sampling.shape
    out_up_sampling_para_pad = util.create_X_pad(out_up_sampling_para, d, no_channel)
    out_conv_2, out_conv_2_para = util.conv(out_up_sampling, out_up_sampling_para_pad, kernel_2)

    _, d, _, no_channel = out_conv_2.shape

    out_conv_2 = out_conv_2 + bias_2

    for i in range(d):
        for j in range(d):
            for k in range(no_channel):
                out_conv_2_para[0][i][j][k][1] = out_conv_2_para[0][i][j][k][1] + bias_2[k]

    out_conv_2 = util.sigmoid(out_conv_2)
    output = out_conv_2

    for i in range(d):
        for j in range(d):
            for k in range(no_channel):
                pT = out_conv_2_para[0][i][j][k][0]
                q = out_conv_2_para[0][i][j][k][1]

                val = np.dot(pT, X_vec)[0][0] + q
                val = util.sigmoid(val)

                if val <= 0.5:
                    global_list_ineq.append([pT, q])
                else:
                    global_list_ineq.append([-pT, -q])

    output = output.flatten()
    binary_vec = []

    for each_e in output:
        if each_e <= 0.5:
            binary_vec.append(0)
        else:
            binary_vec.append(1)

    x = X_vec

    eta, etaTx = util.construct_test_statistic(x, binary_vec, d * d)
    u, v = util.compute_u_v(x, eta, d * d)

    Vminus = np.NINF
    Vplus = np.Inf

    for element in global_list_ineq:
        aT = element[0]
        b = element[1]

        a_scalar = np.dot(aT, v)[0][0]
        b_scalar = np.dot(aT, u)[0][0] + b

        if a_scalar == 0:
            if b > 0:
                print('Error B')

        elif a_scalar > 0:
            Vplus = min(Vplus, -b_scalar / a_scalar)
        else:
            Vminus = max(Vminus, -b_scalar / a_scalar)

    return Vplus - Vminus


from mpi4py import MPI
COMM = MPI.COMM_WORLD

start_time = None

if COMM.rank == 0:
    start_time = time.time()

    max_iteration = 120
    no_thread = COMM.size

    iter_each_thread = int(max_iteration / no_thread)

else:
    iter_each_thread = None

iter_each_thread = COMM.bcast(iter_each_thread, root=0)

local_list_length = []

for i in range(iter_each_thread):

    length = run()

    if length is not None:
        local_list_length.append(length)


total_list_length = COMM.gather(local_list_length, root=0)

if COMM.rank == 0:
    total_list_length = [_i for temp in total_list_length for _i in temp]

    print(total_list_length)

    print("--- %s seconds ---" % (time.time() - start_time))