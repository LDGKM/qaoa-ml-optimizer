"""Stratégies d'initialisation des angles QAOA — le cœur du projet.

Une bonne initialisation réduit le nombre d'appels au circuit quantique (coûteux et bruités
sur du vrai matériel) et évite les mauvais minima locaux.

- random      : angles tirés au hasard (référence naïve) ;
- ramp        : rampe linéaire inspirée du calcul adiabatique (heuristique sans apprentissage) ;
- mean        : moyenne des angles optimaux des graphes d'entraînement (« concentration des
                paramètres » : les angles optimaux varient peu d'un graphe à l'autre) ;
- ml          : un modèle d'apprentissage prédit les angles à partir des caractéristiques du graphe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .graphs import FEATURES, graph_features


def random_init(p: int, rng: np.random.Generator) -> np.ndarray:
    return np.concatenate([rng.uniform(0, np.pi, p), rng.uniform(0, np.pi / 2, p)])


def ramp_init(p: int, delta: float = 0.75) -> np.ndarray:
    """gamma croît et |beta| décroît linéairement, comme une évolution adiabatique discrétisée.

    Le signe négatif de beta vient de la convention de PennyLane (H_C = -C) : avec
    beta > 0, on part dans la mauvaise direction et le ratio initial tombe sous le hasard.
    """
    t = (np.arange(p) + 0.5) / p
    return np.concatenate([delta * t, -delta * (1 - t)])


class InitPredictor:
    """Prédit les angles optimaux d'un graphe à partir de ses caractéristiques.

    kind : "mlp" (réseau de neurones), "rf" (forêt aléatoire), "knn" (plus proches voisins)
           ou "mean" (prédit toujours la moyenne : la référence à battre).
    """

    def __init__(self, kind: str = "rf", seed: int = 0):
        self.kind = kind
        self.seed = seed

    def _make(self):
        if self.kind == "mlp":
            reg = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=3000, random_state=self.seed)
        elif self.kind == "rf":
            reg = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=self.seed)
        elif self.kind == "knn":
            reg = KNeighborsRegressor(n_neighbors=5, weights="distance")
        else:
            raise ValueError(self.kind)
        return make_pipeline(StandardScaler(), reg)

    def fit(self, df: pd.DataFrame) -> "InitPredictor":
        """df : sortie de dataset.build_dataset (colonnes FEATURES + param_0..param_{2p-1})."""
        self.param_cols_ = [c for c in df.columns if c.startswith("param_")]
        Y = df[self.param_cols_].to_numpy()
        self.mean_ = Y.mean(axis=0)
        if self.kind != "mean":
            self.model_ = self._make().fit(df[FEATURES].to_numpy(), Y)
        return self

    def predict_params(self, graph) -> np.ndarray:
        if self.kind == "mean":
            return self.mean_.copy()
        x = pd.DataFrame([graph_features(graph)])[FEATURES].to_numpy()
        return self.model_.predict(x)[0]
