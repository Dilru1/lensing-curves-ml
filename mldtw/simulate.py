import numpy as np
import VBMicrolensing

vbm = VBMicrolensing.VBMicrolensing()

T = np.arange(0, 10, 1 / 48)
SIGMA = 0.01
CLASSES = ["constant", "variable", "stellar event", "FFP event", "flare"]


def magnification(u):
    return (u**2 + 2) / (u * np.sqrt(u**2 + 4))


def constant(rng):
    return np.ones_like(T), {}


def variable(rng):
    P = 10 ** rng.uniform(np.log10(0.2), np.log10(5))
    amp = 10 ** rng.uniform(-1.5, -0.5)
    w = 2 * np.pi * T / P + rng.uniform(0, 2 * np.pi)
    return 1 + amp * (np.sin(w) + 0.3 * np.sin(2 * w) + 0.1 * np.sin(3 * w)), dict(P=P, amp=amp)


def stellar_event(rng):
    p = dict(t0=rng.uniform(3, 7), tE=10 ** rng.uniform(np.log10(3), np.log10(60)),
             u0=rng.uniform(0.02, 0.6), fs=rng.uniform(0.3, 1.0))
    u = np.sqrt(p["u0"] ** 2 + ((T - p["t0"]) / p["tE"]) ** 2)
    return p["fs"] * magnification(u) + 1 - p["fs"], p


def ffp_event(rng):
    rho = 10 ** rng.uniform(-1, np.log10(3))
    p = dict(t0=rng.uniform(3, 7), tE=10 ** rng.uniform(np.log10(0.05), np.log10(0.5)),
             u0=rng.uniform(0, max(0.5, 0.8 * rho)), rho=rho, fs=rng.uniform(0.5, 1.0))
    u = np.sqrt(p["u0"] ** 2 + ((T - p["t0"]) / p["tE"]) ** 2)
    A = np.array([vbm.ESPLMag2(float(ui), float(rho)) for ui in u])
    return p["fs"] * A + 1 - p["fs"], p


def flare(rng):
    t0, t_half = rng.uniform(3, 7), 10 ** rng.uniform(-2, np.log10(0.2))
    amp = 10 ** rng.uniform(np.log10(0.05), 0)
    x = (T - t0) / t_half
    f = np.zeros_like(T)
    rise, decay = (x > -1) & (x <= 0), x > 0
    f[rise] = 1 + 1.941 * x[rise] - 0.175 * x[rise] ** 2 - 2.246 * x[rise] ** 3 - 1.125 * x[rise] ** 4
    f[decay] = 0.6890 * np.exp(-1.600 * x[decay]) + 0.3030 * np.exp(-0.2783 * x[decay])
    return 1 + amp * np.clip(f, 0, None), dict(t0=t0, t_half=t_half, amp=amp)


GENERATORS = [constant, variable, stellar_event, ffp_event, flare]


def make_dataset(n_per_class, seed=0, classes=range(5)):
    rng = np.random.default_rng(seed)
    X, y, params = [], [], []
    for k in classes:
        made = 0
        while made < n_per_class:
            f, p = GENERATORS[k](rng)
            if k >= 2 and f.max() - 1 < 5 * SIGMA:
                continue
            X.append(f + rng.normal(0, SIGMA, T.size))
            y.append(k)
            params.append(p)
            made += 1
    return np.array(X) - 1, np.array(y), params


def centre(X):
    m = X.shape[1] // 2
    return np.array([np.roll(x, m - x.argmax()) for x in X])
