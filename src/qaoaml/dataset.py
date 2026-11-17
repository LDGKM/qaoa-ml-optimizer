"""Construction du jeu de données d'apprentissage : graphe -> angles QAOA optimaux.

Pour chaque graphe aléatoire, on cherche les meilleurs angles (départ en rampe + quelques
départs aléatoires) et on enregistre caractéristiques du graphe et angles canonisés.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import networkx as nx
import numpy as np
import pandas as pd

from .graphs import graph_features, random_graph
from .init import random_init, ramp_init
from .qaoa import QAOASimulator, canonicalize, optimize


def best_params(g: nx.Graph, p: int, restarts: int = 3, seed: int = 0):
    """Meilleurs angles trouvés pour g. On privilégie la solution issue de la rampe
    (famille de solutions « lisse », plus facile à apprendre) sauf si un départ aléatoire
    fait nettement mieux."""
    rng = np.random.default_rng(seed)
    sim = QAOASimulator(g)
    best = optimize(sim, ramp_init(p))
    for _ in range(restarts):
        r = optimize(sim, random_init(p, rng))
        if r.approx_ratio > best.approx_ratio + 1e-3:
            best = r
    return canonicalize(best.params), best.approx_ratio


def _one(args):
    seed, p, n_min, n_max, restarts = args
    rng = np.random.default_rng(seed)
    g = random_graph(rng, n_min, n_max)
    params, ratio = best_params(g, p, restarts, seed)
    row = {"seed": seed, **graph_features(g), "opt_ratio": ratio,
           "edges": " ".join(f"{i}-{j}" for i, j in g.edges())}
    row.update({f"param_{k}": v for k, v in enumerate(params)})
    return row


def build_dataset(n_graphs: int = 200, p: int = 2, n_min: int = 6, n_max: int = 10,
                  restarts: int = 3, seed: int = 0, workers: int | None = None) -> pd.DataFrame:
    jobs = [(seed * 100_000 + i, p, n_min, n_max, restarts) for i in range(n_graphs)]
    if workers == 1:
        rows = [_one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(_one, jobs, chunksize=4))
    return pd.DataFrame(rows)


def graph_from_row(row) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(int(row["n_nodes"])))
    g.add_edges_from(tuple(map(int, e.split("-"))) for e in str(row["edges"]).split())
    return g
