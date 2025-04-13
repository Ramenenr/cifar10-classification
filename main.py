# 加载数据
from data_loader import load_cifar10
from neural_network import nn
from train_test import train, test
from hyperparameter_search import hyperparameter_search
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

X_train, y_train, X_test, y_test = load_cifar10('cifar-10-batches-py')

# 展平数据
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# 预处理数据，零均值化
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_test -= mean_image

# 划分训练集和验证集
num_training = 49000
num_validation = 1000
X_val = X_train[num_training:num_training + num_validation]
y_val = y_train[num_training:num_training + num_validation]
X_train = X_train[:num_training]
y_train = y_train[:num_training]

# 参数查找
# best_model = hyperparameter_search(X_train, y_train, X_val, y_val)

# 利用表现最优的超参数重新训练模型
best_model = nn(input_size=3 * 32 * 32, hidden_size=128, output_size=10)
stats = train(best_model, X_train, y_train, X_val, y_val, learning_rate=1e-3, reg=1e-4, num_iters=50 * 50000 // 256, batch_size=256)

# 测试模型
test_accuracy = test(best_model, X_test, y_test)
print('Test accuracy: %f' % test_accuracy)


# 保存模型权重
# np.savez('model_weights.npz', W1=best_model.W1, b1=best_model.b1, W2=best_model.W2, b2=best_model.b2)