import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import time

import gen_data
import util
import parametric_si


def run():
    d = 8
    IMG_WIDTH = d
    IMG_HEIGHT = d
    IMG_CHANNELS = 1
    mu_1 = 0
    mu_2 = 2

    threshold = 20

    # np.random.seed(1)
    X_test, Y_test = gen_data.generate(1, IMG_WIDTH, mu_1, mu_2)

    model = load_model('./model/test_' + str(d) + '.h5')

    output = model.predict(X_test, verbose=1)

    output = output.flatten()
    binary_vec = []

    for each_e in output:
        if each_e <= 0.5:
            binary_vec.append(0)
        else:
            binary_vec.append(1)

    # print("Observe",  binary_vec)

    X_vec = (X_test.flatten()).reshape((d * d, 1))
    x_obs = X_vec

    eta, etaTx = util.construct_test_statistic(x_obs, binary_vec, d * d)
    u, v = util.compute_u_v(x_obs, eta, d * d)

    list_zk, list_results = parametric_si.run_parametric_si(u, v, model, d, IMG_CHANNELS, threshold)

    z_interval = util.construct_z(binary_vec, list_zk, list_results)

    length = 0
    for interval in z_interval:
        length = length + (interval[1] - interval[0])

    # print(length)

    return length


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