"""QAOA (Quantum Approximate Optimization Algorithm) pour MaxCut.

Le circuit QAOA de profondeur p prépare
    |gamma, beta> = U_M(beta_p) U_C(gamma_p) ... U_M(beta_1) U_C(gamma_1) |+>^n
avec U_C(gamma) = exp(-i gamma H_C) (opérateur de coût) et U_M(beta) = exp(-i beta sum X_i) (mélangeur).
Un optimiseur classique règle les 2p angles pour maximiser la coupe moyenne mesurée.

Deux implémentations, qui donnent exactement les mêmes résultats :
- `pennylane_circuit` : le vrai circuit quantique (lisible, exécutable sur matériel) ;
- `QAOASimulator`     : simulation numpy spécialisée, ~100x plus rapide, utilisée pour
                         générer des milliers d'optimisations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from scipy.optimize import minimize

from .maxcut import bit_table, cut_values


class QAOASimulator:
    """Simulateur exact du QAOA-MaxCut par vecteur d'état.

    U_C est diagonal (phase exp(i gamma C(z)) sur chaque état de base z) ;
    U_M applique RX(2 beta) sur chaque qubit.
    """

    def __init__(self, g: nx.Graph):
        self.g = g
        self.n = g.number_of_nodes()
        self.cuts = cut_values(g).astype(float)
        self.c_max = self.cuts.max()

    def state(self, gammas, betas) -> np.ndarray:
        n = self.n
        psi = np.full(2**n, 2 ** (-n / 2), dtype=complex)
        for gamma, beta in zip(gammas, betas):
            psi = psi * np.exp(1j * gamma * self.cuts)  # = exp(-i gamma H_C), H_C = -C (convention PennyLane)
            psi = psi.reshape([2] * n)
            c, s = np.cos(beta), np.sin(beta)
            for axis in range(n):
                psi = c * psi - 1j * s * np.flip(psi, axis=axis)
            psi = psi.reshape(-1)
        return psi

    def probabilities(self, params) -> np.ndarray:
        p = len(params) // 2
        return np.abs(self.state(params[:p], params[p:])) ** 2

    def expected_cut(self, params) -> float:
        return float(self.probabilities(params) @ self.cuts)

    def approx_ratio(self, params) -> float:
        """Coupe moyenne / coupe maximale : 1 = parfait, ~0.5 = hasard."""
        return self.expected_cut(params) / self.c_max

    def most_likely(self, params, k: int = 5):
        """Les k partitions les plus probables : (bits, probabilité, valeur de la coupe)."""
        probs = self.probabilities(params)
        idx = np.argsort(probs)[::-1][:k]
        bits = bit_table(self.n)
        return [(bits[z], float(probs[z]), int(self.cuts[z])) for z in idx]


@dataclass
class OptimizationResult:
    params: np.ndarray
    approx_ratio: float
    nfev: int  # nombre d'évaluations du circuit (gradients inclus) : le vrai coût sur machine quantique
    history: list = field(default_factory=list)


def optimize(sim: QAOASimulator, init, maxiter: int = 300, tol: float = 1e-6) -> OptimizationResult:
    """Optimise les angles depuis `init` (L-BFGS-B, gradients par différences finies)."""
    history = []

    def f(x):
        val = -sim.expected_cut(x)
        history.append(-val / sim.c_max)
        return val

    res = minimize(f, np.asarray(init, dtype=float), method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": tol * 1e-3, "gtol": tol})
    return OptimizationResult(res.x, sim.approx_ratio(res.x), int(res.nfev), history)


def canonicalize(params) -> np.ndarray:
    """Ramène les angles dans une zone de référence.

    Pour MaxCut non pondéré, l'énergie est invariante par gamma -> gamma + 2 pi,
    beta -> beta + pi/2 et (gamma, beta) -> (-gamma, -beta). Sans cette étape, deux
    solutions équivalentes auraient des coordonnées différentes, ce qui brouillerait l'apprentissage.
    """
    params = np.asarray(params, dtype=float).copy()
    p = len(params) // 2
    g = (params[:p] + np.pi) % (2 * np.pi) - np.pi
    b = (params[p:] + np.pi / 4) % (np.pi / 2) - np.pi / 4
    if g.mean() < 0:
        g, b = -g, -b
    return np.concatenate([g, b])


def pennylane_circuit(g: nx.Graph, p: int, device: str = "default.qubit", shots=None):
    """Le circuit QAOA écrit avec PennyLane. Renvoie (qnode, hamiltonien de coût).

    qnode(params) renvoie <H_C> = -(coupe moyenne).
    """
    import pennylane as qml

    n = g.number_of_nodes()
    cost_h, mixer_h = qml.qaoa.maxcut(g)
    dev = qml.device(device, wires=n, shots=shots)

    @qml.qnode(dev)
    def circuit(params):
        for w in range(n):
            qml.Hadamard(wires=w)
        for k in range(p):
            qml.qaoa.cost_layer(params[k], cost_h)
            qml.qaoa.mixer_layer(params[p + k], mixer_h)
        return qml.expval(cost_h)

    return circuit, cost_h
