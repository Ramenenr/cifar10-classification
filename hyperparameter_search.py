def hyperparameter_search(X_train, y_train, X_val, y_val):
    # 超参数包含学习率、隐藏层大小、正则化权重、迭代次数和batch size
    learning_rates = [1e-3, 1e-4]
    hidden_sizes = [64, 128]
    reg_strengths = [1e-3, 1e-4]
    batch_sizes = [128, 256]
    num_epochs_list = [10, 30, 50]

    results = {}
    best_val_acc = 0
    best_model = None

    num_train = X_train.shape[0]

    for lr in learning_rates:
        for hs in hidden_sizes:
            for reg in reg_strengths:
                for bs in batch_sizes:
                    for num_epochs in num_epochs_list:
                        iterations_per_epoch = max(num_train // bs, 1)
                        num_iters = num_epochs * iterations_per_epoch  # 计算 num_iters 为 epoch 的整数倍

                        model = nn(input_size=3 * 32 * 32, hidden_size=hs, output_size=10)
                        stats = train(model, X_train, y_train, X_val, y_val, learning_rate=lr, reg=reg,
                                      num_iters=num_iters, batch_size=bs)
                        val_acc = stats['val_acc_history'][-1]

                        results[(lr, hs, reg, bs, num_epochs)] = val_acc

                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            best_model = model

    for lr, hs, reg, bs, num_epochs in sorted(results):
        val_acc = results[(lr, hs, reg, bs, num_epochs)]
        print('lr %e hs %d reg %e bs %d num_epochs %d val accuracy: %f' % (
            lr, hs, reg, bs, num_epochs, val_acc))

    print('best validation accuracy achieved during cross - validation: %f' % best_val_acc)
    return best_model    