"""Génération de graphes et calcul de leurs caractéristiques.

Les caractéristiques (« features ») décrivent la structure d'un graphe avec quelques nombres :
c'est ce que le modèle d'apprentissage reçoit en entrée pour prédire de bons paramètres QAOA.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

FEATURES = [
    "n_nodes", "n_edges", "density", "deg_mean", "deg_std", "deg_min", "deg_max",
    "clustering", "triangles_per_node", "lap_max", "lap_fiedler", "bipartite",
]


def random_graph(rng: np.random.Generator, n_min: int = 6, n_max: int = 10) -> nx.Graph:
    """Graphe aléatoire : Erdős–Rényi (densité variable) ou régulier (degré 3 ou 4)."""
    while True:
        n = int(rng.integers(n_min, n_max + 1))
        if rng.random() < 0.5:
            g = nx.gnp_random_graph(n, float(rng.uniform(0.25, 0.75)), seed=int(rng.integers(1 << 31)))
        else:
            d = int(rng.choice([3, 4]))
            if (n * d) % 2:
                continue
            g = nx.random_regular_graph(d, n, seed=int(rng.integers(1 << 31)))
        if g.number_of_edges() >= 2:
            return nx.convert_node_labels_to_integers(g)


def graph_features(g: nx.Graph) -> dict:
    n, m = g.number_of_nodes(), g.number_of_edges()
    deg = np.array([d for _, d in g.degree()], dtype=float)
    lap = np.sort(np.linalg.eigvalsh(nx.laplacian_matrix(g).toarray().astype(float)))
    return {
        "n_nodes": n,
        "n_edges": m,
        "density": nx.density(g),
        "deg_mean": deg.mean(),
        "deg_std": deg.std(),
        "deg_min": deg.min(),
        "deg_max": deg.max(),
        "clustering": nx.average_clustering(g),
        "triangles_per_node": sum(nx.triangles(g).values()) / 3 / n,
        "lap_max": lap[-1],
        "lap_fiedler": lap[1] if n > 1 else 0.0,
        "bipartite": float(nx.is_bipartite(g)),
    }
