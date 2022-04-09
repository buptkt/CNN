from Solver import *
from load_data import *
from cnn import *
import numpy as np


if __name__ == '__main__':
    data = {}
    X_train, y_train = load_mnist('.\\mnist', 'train')
    X_train = X_train.astype(np.float32)
    # mask = np.random.choice(X_train.shape[0], 100)
    # data['X_train'], data['y_train'] = X_train[mask], y_train[mask]
    data['X_train'], data['y_train'] = X_train, y_train
    # 数据预处理，均值化
    data['X_train'] -= np.mean(data['X_train'], axis=0)

    data['X_val'], data['y_val'] = load_mnist('.\\mnist', 't10k')

    model = ThreeLayerConvNet(input_dim=(1, 28, 28), num_filters=7, filter_size=5, hidden_dim=50, reg=0.5, weight_scale=1e-3)
    solver = Solver(model=model, data=data,
                    num_epochs=20,
                    batch_size=64,
                    update_rule='adam',
                    optim_config={'learning_rate': 1e-4, },
                    lr_decay=0.99,
                    verbose=True,
                    print_every=1000,
                    num_val_samples=10000
                    )
    solver.train()
