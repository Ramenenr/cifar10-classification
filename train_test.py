import numpy as np
import matplotlib.pyplot as plt


# 模型训练
def train(model, X, y, X_val, y_val, learning_rate, reg, num_iters, batch_size, verbose=False, learning_rate_decay=0.95):
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train // batch_size, 1)

    # 记录训练集和验证集损失，以及最优参数和accuracy
    best_val_acc = 0
    best_params = {}
    loss_history = []
    train_acc_history = []
    val_acc_history = []
    val_loss_history = []

    for it in range(num_iters):
        indices = np.random.choice(num_train, batch_size)
        X_batch = X[indices]
        y_batch = y[indices]

        loss, grads = model.loss(X_batch, y_batch, reg)
        loss_history.append(loss)

        model.W1 += -learning_rate * grads['W1']
        model.b1 += -learning_rate * grads['b1']
        model.W2 += -learning_rate * grads['W2']
        model.b2 += -learning_rate * grads['b2']

        if verbose and it % 100 == 0:
            print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        if it % iterations_per_epoch == 0:
            train_acc = (model.forward(X_batch).argmax(axis=1) == y_batch).mean()
            val_acc = (model.forward(X_val).argmax(axis=1) == y_val).mean()
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

            val_loss, _ = model.loss(X_val, y_val, reg)
            val_loss_history.append(val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = {
                    'W1': model.W1.copy(),
                    'b1': model.b1.copy(),
                    'W2': model.W2.copy(),
                    'b2': model.b2.copy()
                }

            learning_rate *= learning_rate_decay

    model.W1 = best_params['W1']
    model.b1 = best_params['b1']
    model.W2 = best_params['W2']
    model.b2 = best_params['b2']

    # 可视化训练过程
    plt.figure(figsize=(12, 8))

    # 训练集损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    # 验证集损失曲线
    plt.subplot(2, 2, 3)
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.title('Validation Loss')
    plt.legend()

    # 验证集的准确率曲线
    plt.subplot(2, 2, 4)
    plt.plot(val_acc_history, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return {
        'loss_history': loss_history,
        'train_acc_history': train_acc_history,
        'val_acc_history': val_acc_history,
        'val_loss_history': val_loss_history
    }

# 模型测试
def test(model, X_test, y_test):
    scores = model.forward(X_test)
    y_pred = scores.argmax(axis=1)
    accuracy = (y_pred == y_test).mean()
    return accuracy    