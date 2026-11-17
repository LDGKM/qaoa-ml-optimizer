"""Tests rapides : exactitude du simulateur, solveurs classiques, chaîne complète."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qaoaml import (  # noqa: E402
    InitPredictor, QAOASimulator, brute_force, build_dataset, canonicalize, evaluate, local_search,
    optimize, pennylane_circuit, random_graph, ramp_init,
)


@pytest.fixture
def graph():
    return random_graph(np.random.default_rng(7), 6, 8)


def test_simulator_matches_pennylane(graph):
    """Le simulateur rapide et le vrai circuit PennyLane donnent la même coupe moyenne."""
    x = np.array([0.4, 0.9, -0.3, -0.2])
    circuit, _ = pennylane_circuit(graph, p=2)
    assert np.isclose(QAOASimulator(graph).expected_cut(x), -float(circuit(x)), atol=1e-8)


def test_uniform_superposition_gives_half_the_edges(graph):
    """Angles nuls : état |+>^n, chaque arête est coupée avec probabilité 1/2."""
    assert np.isclose(QAOASimulator(graph).expected_cut(np.zeros(2)), graph.number_of_edges() / 2)


def test_classical_solvers(graph):
    c_max, part = brute_force(graph)
    assert c_max == sum(part[i] != part[j] for i, j in graph.edges())
    assert local_search(graph)[0] <= c_max


def test_canonicalize_preserves_energy(graph):
    sim = QAOASimulator(graph)
    x = np.array([5.0, -2.0, 1.3, -0.9])
    assert np.isclose(sim.expected_cut(x), sim.expected_cut(canonicalize(x)))


def test_optimization_improves_ramp(graph):
    sim = QAOASimulator(graph)
    res = optimize(sim, ramp_init(2))
    assert res.approx_ratio >= sim.approx_ratio(ramp_init(2)) > 0.6


def test_learned_initialization_pipeline():
    train = build_dataset(n_graphs=20, p=1, n_min=5, n_max=7, restarts=1, seed=0, workers=1)
    test = build_dataset(n_graphs=4, p=1, n_min=5, n_max=7, restarts=1, seed=1, workers=1)
    res = evaluate(train, test, strategies=["ramp", "knn"])
    assert set(res.strategy) == {"ramp", "knn"}
    assert (res.final_ratio > 0.6).all()
    assert len(InitPredictor("rf").fit(train).predict_params(random_graph(np.random.default_rng(0)))) == 2
