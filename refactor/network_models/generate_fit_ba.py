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
gam = 3
g = nx.barabasi_albert_graph(N, gam)


def powerlaw(x, gam):
    return np.power(x, -gam)


def ba_distrib(x, m):
    return 2 * (m**2) * np.power(x, -3)


def logbins(deg, counts):
    max_exp = int(np.log2(np.max(deg)))
    bins = np.power(2.0, np.arange(0, max_exp + 1))
    bins_counts = np.zeros_like(bins)
    total_deg = np.zeros_like(bins)
    for d, c in zip(deg, counts):
        idx = 0
        for i in range(len(bins)):
            if d >= bins[i]:
                idx = i
        bins_counts[idx] += c
        total_deg[idx] += c * d
    distrib = np.divide(bins_counts, bins)
    degs = np.divide(total_deg, bins_counts)
    return degs, distrib


nx.write_gml(g, "classic/ba.gml")
degree_sequence = sorted([d for n, d in g.degree()], reverse=True)
deg, counts = np.unique(degree_sequence, return_counts=True)
counts = counts / np.sum(counts)
log_deg, log_count = logbins(deg, counts)
x = np.array(range(1, np.max(deg)), dtype=np.float64)
plt.scatter(log_deg, log_count, c="grey", label="Degrees")
plt.plot(x, ba_distrib(x, gam), c="#f76468", label="BA Distribution")
plt.xlabel("degree")
plt.ylabel("P(degree)")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.show()
