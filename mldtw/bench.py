"""Compare time-series distances on light curves with nearest-neighbour methods."""
import time
import numpy as np
from scipy.spatial.distance import cdist
from tslearn.metrics import cdist_dtw, cdist_soft_dtw


def centre(X):
    """Shift each curve so its peak is mid-window (removes t0, keeps tE)."""
    m = X.shape[1] // 2
    return np.array([np.roll(x, m - x.argmax()) for x in X])


class Bench:
    def __init__(self, X_train, X_test, radius=24):
        A, B = X_train - 1, X_test - 1                       # flux excess
        Ac, Bc = centre(A), centre(B)
        band = dict(global_constraint="sakoe_chiba", sakoe_chiba_radius=radius, n_jobs=-1)
        self.metrics = {
            "Euclid":             lambda: cdist(B, A),
            "Euclid, centred":    lambda: cdist(Bc, Ac),
            "DTW":                lambda: cdist_dtw(B, A, n_jobs=-1),
            "DTW band":           lambda: cdist_dtw(B, A, **band),
            "Centred + DTW":      lambda: cdist_dtw(Bc, Ac, n_jobs=-1),
            "Centred + DTW band": lambda: cdist_dtw(Bc, Ac, **band),
        }
        self.soft = lambda g: cdist_soft_dtw(Bc, Ac, gamma=g)  # centred soft-DTW
        self.D = {}

    def run(self, names=None):
        for n in names or self.metrics:
            t = time.time()
            self.D[n] = self.metrics[n]()
            print(f"{n:20s} {time.time() - t:5.1f} s")
        return self

    def add(self, name, D):
        self.D[name] = D
        return self

    def classify(self, y_train, y_test):
        """1-NN accuracy for each distance."""
        return {n: round(float(np.mean(y_train[d.argmin(1)] == y_test)), 3) for n, d in self.D.items()}

    def regress(self, t_train, t_test, k=5):
        """k-NN regression RMSE for each distance; also stores predictions."""
        self.pred = {n: t_train[np.argsort(d, 1)[:, :k]].mean(1) for n, d in self.D.items()}
        return {n: round(float(np.sqrt(np.mean((p - t_test) ** 2))), 3) for n, p in self.pred.items()}
