import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import time

import gen_data
import util
import parametric_si


def run():
    n = 16

    d = int(np.sqrt(n))
    IMG_WIDTH = d
    mu_1 = 0
    mu_2 = 0

    X_test, Y_test = gen_data.generate(1, IMG_WIDTH, mu_1, mu_2)

    model = load_model('./model/test_' + str(d) + '.h5')

    output = model.predict(X_test, verbose=0)

    output = output.flatten()
    X_vec = X_test.flatten()

    m_a = 0
    m_b = 0
    n_a = 0
    n_b = 0

    for i in range(len(output)):
        if output[i] <= 0.5:
            n_a = n_a + 1
            m_a = m_a + X_vec[i]
        else:
            n_b = n_b + 1
            m_b = m_b + X_vec[i]

    if (n_a == 0) or (n_b == 0):
        return None

    m_a = m_a / n_a
    m_b = m_b / n_b

    test_statistic = m_a - m_b

    pivot = util.compute_naive_p(test_statistic, n_a, n_b, 1)

    return pivot


from mpi4py import MPI
COMM = MPI.COMM_WORLD

start_time = None

if COMM.rank == 0:
    start_time = time.time()

    max_iteration = 200
    no_thread = COMM.size

    iter_each_thread = int(max_iteration / no_thread)

else:
    iter_each_thread = None

iter_each_thread = COMM.bcast(iter_each_thread, root=0)

local_list_pivot = []

for i in range(iter_each_thread):

    pivot = run()

    if pivot is not None:
        local_list_pivot.append(pivot)


total_list_pivot = COMM.gather(local_list_pivot, root=0)

if COMM.rank == 0:
    total_list_pivot = [_i for temp in total_list_pivot for _i in temp]

    detect = 0
    reject = 0

    for pivot in total_list_pivot:
        if pivot is not None:
            detect = detect + 1
            if pivot < 0.05:
                reject = reject + 1

    print(reject, detect, reject / detect)

    print("--- %s seconds ---" % (time.time() - start_time))