import numpy as np
from scipy.stats import skewnorm


def generate_non_normal(n, d, mu_1, mu_2):
    list_X_train = []
    list_X_label = []

    for _ in range(n):
        X_train = []
        X_label = []

        for i in range(d):
            if (i < d / 4) or (i >= 3 * d / 4):
                vec_train = []
                vec_label = []
                gen_vec = list(np.random.normal(mu_1, 1, d))
                # gen_vec = list(np.random.laplace(mu_1, 1, d))
                # gen_vec = list(skewnorm.rvs(a=10, loc=mu_1, scale=1, size=d))
                # gen_vec = list(np.random.standard_t(20, d))

                for j in range(d):
                    vec_train.append([gen_vec[j]])
                    vec_label.append([False])

                X_train.append(list(vec_train))
                X_label.append(list(vec_label))
            else:
                vec_train = []
                vec_label = []

                for j in range(d):
                    if (j < d / 4) or (j >= 3 * d / 4):
                        vec_train.append([float(np.random.normal(mu_1, 1))])
                        # vec_train.append([float(np.random.laplace(mu_1, 1))])
                        # vec_train.append([float(skewnorm.rvs(a=10, loc=mu_1, scale=1))])
                        # vec_train.append([float(np.random.standard_t(20, 1))])

                        vec_label.append([False])
                    else:
                        vec_train.append([float(np.random.normal(mu_2, 1))])
                        # vec_train.append([float(np.random.laplace(mu_2, 1))])
                        # vec_train.append([float(skewnorm.rvs(a=10, loc=mu_2, scale=1))])
                        # vec_train.append([float(np.random.standard_t(20, 1))])

                        vec_label.append([True])

                X_train.append(list(vec_train))
                X_label.append(list(vec_label))

        list_X_train.append(np.array(X_train))
        list_X_label.append(np.array(X_label))

    return np.array(list_X_train), np.array(list_X_label)


def generate(n, d, mu_1, mu_2):
    list_X_train = []
    list_X_label = []

    for _ in range(n):
        X_train = []
        X_label = []

        for i in range(d):
            if (i < d / 4) or (i >= 3 * d / 4):
                vec_train = []
                vec_label = []
                gen_vec = list(np.random.normal(mu_1, 1, d))

                for j in range(d):
                    vec_train.append([gen_vec[j]])
                    vec_label.append([False])

                X_train.append(list(vec_train))
                X_label.append(list(vec_label))
            else:
                vec_train = []
                vec_label = []

                for j in range(d):
                    if (j < d / 4) or (j >= 3 * d / 4):
                        vec_train.append([float(np.random.normal(mu_1, 1))])
                        vec_label.append([False])
                    else:
                        vec_train.append([float(np.random.normal(mu_2, 1))])
                        vec_label.append([True])

                X_train.append(list(vec_train))
                X_label.append(list(vec_label))

        list_X_train.append(np.array(X_train))
        list_X_label.append(np.array(X_label))

    return np.array(list_X_train), np.array(list_X_label)


if __name__ == '__main__':
    list_X_train, list_X_label = generate(2, 8, 2, 8)

    print(list_X_label[0])
    print(list_X_label[1])
