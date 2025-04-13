import numpy as np
import os
import pickle


def load_cifar10_batch(file_path):
    with open(file_path, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_cifar10(root):
    # 加载整个 CIFAR - 10 数据集
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(root, 'data_batch_%d' % (b,))
        X, Y = load_cifar10_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_cifar10_batch(os.path.join(root, 'test_batch'))
    return Xtr, Ytr, Xte, Yte
    # 输入(50000, 32, 32, 3)
    # 输出 label 种类10    