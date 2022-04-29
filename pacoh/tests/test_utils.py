import numpy as np


def get_simple_sinusoid_dataset():
    n_train = 50
    x_train = np.random.uniform(-4, 4, size=(n_train, 1))
    y_train = np.sin(x_train) + np.random.normal(scale=0.1, size=x_train.shape)

    n_val = 200

    x_plot = np.linspace(-6, 6, num=n_val)
    x_plot = np.expand_dims(x_plot, -1)
    y_val = np.sin(x_plot) + np.random.normal(scale=0.1, size=x_plot.shape)

    return x_train, y_train, x_plot, y_val
