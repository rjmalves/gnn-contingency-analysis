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
N = 1000
p = 0.1
g = nx.erdos_renyi_graph(N, p)


def binomial(x, N, p):
    return np.multiply(
        np.multiply(binom(N - 1, x), np.power(p, x)),
        np.power(1 - p, [N - 1 - x]),
    ).flatten()


nx.write_gml(g, "classic/gnp.gml")
degree_sequence = sorted([d for n, d in g.degree()], reverse=True)
deg, counts = np.unique(degree_sequence, return_counts=True)
counts = counts / np.sum(counts)
plt.scatter(deg, counts, c="grey", label="Degrees")
x = np.array(range(70, 140))
plt.plot(x, binomial(x, N, p), c="#f76468", label="Binomial")
plt.xlabel("degree")
plt.ylabel("P(degree)")
plt.legend()
plt.show()
