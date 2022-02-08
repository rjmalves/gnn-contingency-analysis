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


df = pd.read_csv(
    "./sin/LINHA_TRANSMISSAO.csv",
    encoding="iso-8859-1",
    sep=";",
)
df["dat_entradaoperacao"] = pd.to_datetime(df["dat_entradaoperacao"])
df["dat_desativacao"] = pd.to_datetime(df["dat_desativacao"])
df = df.loc[df["nom_tipolinha"] != "RAMAL DE LINHA", :]
df["de"] = df["id_estado_terminalde"] + "_" + df["nom_subestacao_de"]
df["para"] = df["id_estado_terminalpara"] + "_" + df["nom_subestacao_para"]


def constroi_grafo_do_df(df: pd.DataFrame) -> nx.Graph:
    lista_arestas = []
    for idx, linha in df.iterrows():
        aresta = (linha["de"], linha["para"])
        lista_arestas.append(aresta)
    lista_final = list(set(lista_arestas))
    return nx.Graph(lista_final)


subestacoes = set(df["de"].unique())
subestacoes.update(set(df["para"].unique()))
subestacoes = list(subestacoes)
lista_arestas = []
for idx, linha in df.iterrows():
    aresta = (linha["de"], linha["para"])
    lista_arestas.append(aresta)


g = nx.MultiGraph(lista_arestas)
nx.write_gml(g, "sin_completo.gml")

lista_final = list(set(lista_arestas))

g = nx.Graph(lista_final)
nx.write_gml(g, "sin_simples.gml")
