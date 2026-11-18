import numpy as np
import pytest


def make_data(n=300, p=30, task="regression", seed=0, noise=0.5):
    """Five nonlinear signals in columns 0-4, the rest noise."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    f = np.sin(np.pi * X[:, 0]) + X[:, 1] ** 2 + 1.5 * np.abs(X[:, 2]) + (X[:, 3] > 0.5) + X[:, 4]
    y = f + noise * rng.standard_normal(n)
    if task == "binary":
        y = (y > np.median(y)).astype(int)
    elif task == "multiclass":
        y = np.digitize(y, np.quantile(y, [1 / 3, 2 / 3]))
    return X, y


@pytest.fixture
def data():
    return make_data()


FAST = {"n_experiments": 8, "max_dummy_multiplier": 3, "max_dummies_entered": 20}
