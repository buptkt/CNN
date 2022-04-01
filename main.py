from Solver import *
from load_data import *
from cnn import *
import numpy as np

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = {}
    X_train, y_train = load_mnist('.\\mnist', 'train')
    X_train = X_train.astype(np.float32)
    mask = np.random.choice(X_train.shape[0], 100)
    data['X_train'], data['y_train'] = X_train[mask], y_train[mask]
    data['X_val'], data['y_val'] = load_mnist('.\\mnist', 't10k')

    model = ThreeLayerConvNet(input_dim=(1, 28, 28), num_filters=5, filter_size=7, hidden_dim=100, reg=0.5)
    solver = Solver(model=model, data=data,
                    num_epochs=10,
                    batch_size=50,
                    update_rule='adam',
                    optim_config={'learning_rate': 1e-3, },
                    verbose=True,
                    print_every=10,
                    num_val_samples=100
                    )
    solver.train()
