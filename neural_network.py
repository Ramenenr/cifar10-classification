import numpy as np

class nn:
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        self.W1 = 0.001 * np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = 0.001 * np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
        self.activation = activation

    # 前向传播
    def forward(self, X):
        # 输入层到隐藏层，经过一次线性变换和激活函数
        if self.activation == 'relu':
            self.h1 = np.maximum(0, np.dot(X, self.W1) + self.b1)
        elif self.activation == 'sigmoid':
            self.h1 = 1 / (1 + np.exp(-(np.dot(X, self.W1) + self.b1)))
        # 隐藏层到输出层
        scores = np.dot(self.h1, self.W2) + self.b2
        return scores

    def loss(self, X, y, reg):
        num_train = X.shape[0]
        scores = self.forward(X)
        # softmax
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # 计算 softmax 损失
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        correct_logprobs = -np.log(probs[range(num_train), y])
        data_loss = np.sum(correct_logprobs) / num_train
        # 正则化损失
        reg_loss = 0.5 * reg * (np.sum(self.W1 * self.W1) + np.sum(self.W2 * self.W2))
        loss = data_loss + reg_loss

        # 反向传播
        dscores = probs
        dscores[range(num_train), y] -= 1
        dscores /= num_train

        dW2 = np.dot(self.h1.T, dscores) + reg * self.W2
        db2 = np.sum(dscores, axis=0, keepdims=True)

        if self.activation == 'relu':
            dh1 = np.dot(dscores, self.W2.T)
            dh1[self.h1 <= 0] = 0
        elif self.activation == 'sigmoid':
            dh1 = np.dot(dscores, self.W2.T) * (self.h1 * (1 - self.h1))

        dW1 = np.dot(X.T, dh1) + reg * self.W1
        db1 = np.sum(dh1, axis=0, keepdims=True)

        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

        return loss, grads    