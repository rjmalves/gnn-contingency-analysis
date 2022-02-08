import networkx as nx
import pandas as pd
import numpy as np
import os
import scipy.optimize as sp
from scipy.special import binom, gammaincc
import matplotlib.pyplot as plt
from datetime import datetime

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"],
    }
)


np.random.seed(10)


g = nx.read_gml("./sin/sin_simple.gml")
degree_sequence = sorted([d for n, d in g.degree()], reverse=True)
deg, counts = np.unique(degree_sequence, return_counts=True)
counts = counts / np.sum(counts)


def exponential(x, lam):
    return lam * np.exp(-lam * x)


def powerlaw(x, gam):
    return np.power(np.array(x, dtype=np.float64), -gam)


def ba_distrib(x, m):
    return 2 * (m**2) * np.power(x, -3)


def power_law_exp_cutoff(x, gam, lam):
    C = lam ** (1 - gam) / gammaincc(1 - gam, lam * np.min(x))
    return C * np.multiply(np.power(x, -gam), np.exp(-lam * x))


def stretched_exponential(x, beta, lam):
    C = beta * lam**beta
    return C * np.multiply(
        np.power(x, beta - 1), np.exp(-np.power(lam * x, beta))
    )


plt.scatter(deg, counts, c="grey", label="Degrees")
params, _ = sp.curve_fit(exponential, deg, counts, bounds=(0, np.inf))
print(f"Exponential: {params}")
plt.plot(deg, exponential(deg, *params), c="#f76468", label=f"Exponential")
params, _ = sp.curve_fit(
    power_law_exp_cutoff, deg, counts, p0=(0.5, 0.5), bounds=(0, np.inf)
)
print(f"Power Law w/ Cutoff: {params}")
plt.plot(
    deg,
    power_law_exp_cutoff(deg, *params),
    c="#f76468",
    label=f"Power Law w/ Exp. Cutoff",
    linestyle="dashed",
)
params, _ = sp.curve_fit(stretched_exponential, deg, counts, bounds=(0, np.inf))
print(f"Stretched exponential: {params}")
plt.plot(
    deg,
    stretched_exponential(deg, *params),
    c="#f76468",
    label=f"Stretched Exponential",
    linestyle="dotted",
)
plt.xlabel("degree")
plt.ylabel("P(degree)")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.show()
