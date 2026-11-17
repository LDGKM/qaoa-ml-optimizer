"""Le problème MaxCut et ses solveurs classiques.

MaxCut : couper les sommets d'un graphe en deux groupes pour maximiser le nombre d'arêtes
qui relient les deux groupes. Problème NP-difficile, mais exact par énumération
pour les petits graphes (2^n partitions).
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def bit_table(n: int) -> np.ndarray:
    """bits[z, i] = valeur du qubit i dans l'état de base z (qubit 0 = bit de poids fort, comme PennyLane)."""
    z = np.arange(2**n)
    return (z[:, None] >> (n - 1 - np.arange(n))) & 1


def cut_values(g: nx.Graph) -> np.ndarray:
    """Valeur de la coupe pour chacune des 2^n partitions (vecteur de taille 2^n)."""
    bits = bit_table(g.number_of_nodes())
    c = np.zeros(len(bits), dtype=np.int64)
    for i, j in g.edges():
        c += bits[:, i] ^ bits[:, j]
    return c


def cut_value(g: nx.Graph, partition) -> int:
    return sum(int(partition[i] != partition[j]) for i, j in g.edges())


def brute_force(g: nx.Graph) -> tuple[int, np.ndarray]:
    """Solution exacte par énumération (raisonnable jusqu'à ~20 sommets)."""
    c = cut_values(g)
    z = int(np.argmax(c))
    return int(c[z]), bit_table(g.number_of_nodes())[z]


def local_search(g: nx.Graph, restarts: int = 10, seed: int = 0) -> tuple[int, np.ndarray]:
    """Heuristique classique : partir d'une partition aléatoire et déplacer un sommet
    tant que cela augmente la coupe. Garde le meilleur résultat sur plusieurs départs."""
    rng = np.random.default_rng(seed)
    n = g.number_of_nodes()
    adj = [list(g.neighbors(v)) for v in range(n)]
    best, best_p = -1, None
    for _ in range(restarts):
        p = rng.integers(0, 2, n)
        improved = True
        while improved:
            improved = False
            for v in rng.permutation(n):
                same = sum(p[u] == p[v] for u in adj[v])
                if same > len(adj[v]) - same:  # déplacer v ajoute (same - other) arêtes coupées
                    p[v] ^= 1
                    improved = True
        val = cut_value(g, p)
        if val > best:
            best, best_p = val, p.copy()
    return best, best_p
