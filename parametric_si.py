import numpy as np
import tensorflow as tf

import util


def run_parametric_si(u, v, model, d, IMG_CHANNELS, threshold):
    zk = -threshold

    list_zk = [zk]
    list_results = []

    while zk < threshold:
        x = u + v * zk

        global_list_ineq = []

        X_test = np.reshape(x, (1, d, d, IMG_CHANNELS))

        X_para, X_vec = util.create_X_para(X_test, d)

        X_para_pad = util.create_X_pad(X_para, d, IMG_CHANNELS)

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

        Vminus = np.NINF
        Vplus = np.Inf

        for element in global_list_ineq:
            aT = element[0]
            b = element[1]

            a_scalar = np.dot(aT, v)[0][0]
            b_scalar = np.dot(aT, u)[0][0] + b

            if a_scalar > 0:
                Vplus = min(Vplus, -b_scalar / a_scalar)

        # zk = Vplus + 0.0001
        zk = Vplus + 0.005

        # print(zk)
        # print(binary_vec)
        # print("===========")

        if zk < threshold:
            list_zk.append(zk)
        else:
            list_zk.append(threshold)

        list_results.append(binary_vec)

    return list_zk, list_results