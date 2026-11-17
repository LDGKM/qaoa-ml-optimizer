"""Comparaison des stratégies d'initialisation sur des graphes jamais vus.

Pour chaque graphe de test et chaque stratégie, on mesure :
- init_ratio  : qualité des angles proposés, SANS aucune optimisation ;
- final_ratio : qualité après optimisation complète ;
- nfev        : nombre d'évaluations du circuit nécessaires (le coût sur machine réelle) ;
- gap         : écart au meilleur ratio connu pour ce graphe.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import build_dataset, graph_from_row
from .init import InitPredictor, ramp_init, random_init
from .maxcut import brute_force, local_search
from .qaoa import QAOASimulator, optimize

STRATEGIES = ["random", "ramp", "mean", "knn", "rf", "mlp"]
LABELS = {
    "random": "Aléatoire", "ramp": "Rampe linéaire", "mean": "Moyenne (concentration)",
    "knn": "ML : k plus proches voisins", "rf": "ML : forêt aléatoire", "mlp": "ML : réseau de neurones",
}


def load_or_build(path: Path, **kwargs) -> pd.DataFrame:
    """Les jeux de données sont mis en cache dans results/ (leur calcul prend une minute)."""
    if path.exists():
        return pd.read_csv(path)
    df = build_dataset(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def evaluate(train: pd.DataFrame, test: pd.DataFrame, strategies=STRATEGIES, seed: int = 0) -> pd.DataFrame:
    p = sum(c.startswith("param_") for c in train.columns) // 2
    predictors = {k: InitPredictor(k, seed).fit(train) for k in strategies if k not in ("random", "ramp")}
    rng = np.random.default_rng(seed)
    rows = []
    for _, row in test.iterrows():
        g = graph_from_row(row)
        sim = QAOASimulator(g)
        for k in strategies:
            if k == "random":
                x0 = random_init(p, rng)
            elif k == "ramp":
                x0 = ramp_init(p)
            else:
                x0 = predictors[k].predict_params(g)
            res = optimize(sim, x0)
            rows.append({
                "graph": row["seed"], "strategy": k, "init_ratio": sim.approx_ratio(x0),
                "final_ratio": res.approx_ratio, "nfev": res.nfev,
                "gap": max(row["opt_ratio"], res.approx_ratio) - res.approx_ratio,
            })
    return pd.DataFrame(rows)


def classical_reference(test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in test.iterrows():
        g = graph_from_row(row)
        t0 = time.perf_counter(); c_max, _ = brute_force(g); t_bf = time.perf_counter() - t0
        t0 = time.perf_counter(); c_ls, _ = local_search(g); t_ls = time.perf_counter() - t0
        rows.append({"graph": row["seed"], "local_search_ratio": c_ls / c_max,
                     "t_bruteforce": t_bf, "t_local_search": t_ls})
    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame) -> pd.DataFrame:
    s = res.groupby("strategy").agg(
        init_ratio=("init_ratio", "mean"),
        final_ratio=("final_ratio", "mean"),
        nfev=("nfev", "mean"),
        success=("gap", lambda g: (g < 1e-3).mean()),
    ).reindex([k for k in STRATEGIES if k in res.strategy.unique()])
    s.insert(0, "stratégie", [LABELS[k] for k in s.index])
    return s.reset_index(drop=True)


def run_benchmark(p: int = 2, n_train: int = 200, n_test: int = 50, n_min: int = 6, n_max: int = 10,
                  seed: int = 0, out_dir: str | Path = "results", verbose: bool = True):
    out = Path(out_dir)
    tag = f"p{p}_n{n_min}-{n_max}"
    if verbose:
        print(f"Jeu d'entraînement ({n_train} graphes)…")
    train = load_or_build(out / f"train_{tag}_{n_train}.csv", n_graphs=n_train, p=p,
                          n_min=n_min, n_max=n_max, seed=seed)
    if verbose:
        print(f"Jeu de test ({n_test} graphes)…")
    test = load_or_build(out / f"test_{tag}_{n_test}.csv", n_graphs=n_test, p=p,
                         n_min=n_min, n_max=n_max, seed=seed + 1)
    if verbose:
        print("Évaluation des stratégies…")
    res = evaluate(train, test, seed=seed)
    summary = summarize(res)
    classical = classical_reference(test)

    res.to_csv(out / f"eval_{tag}.csv", index=False)
    summary.to_csv(out / f"summary_{tag}.csv", index=False)
    summary.to_latex(out / f"summary_{tag}.tex", index=False, float_format="%.3f",
                     header=["Stratégie", "Ratio initial", "Ratio final", "Évaluations", "Succès"])
    _plot(res, out / f"summary_{tag}.png", p)
    return summary, classical, res


def _plot(res: pd.DataFrame, path: Path, p: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [k for k in STRATEGIES if k in res.strategy.unique()]
    colors = ["#c9ced6", "#8d99ae", "#4a5568", "#b9a3d8", "#9b7fc1", "#6a4c93"][: len(order)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    data = [res[res.strategy == k]["init_ratio"] for k in order]
    bp = axes[0].boxplot(data, vert=False, patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    axes[0].set_yticks(range(1, len(order) + 1), [LABELS[k] for k in order])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Ratio d'approximation SANS optimisation")
    nf = res.groupby("strategy")["nfev"].mean().reindex(order)
    axes[1].barh(range(len(order)), nf, color=colors)
    axes[1].set_yticks(range(len(order)), [])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Évaluations du circuit jusqu'à convergence")
    for i, v in enumerate(nf):
        axes[1].text(v, i, f" {v:.0f}", va="center", fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"QAOA p = {p} — graphes de test jamais vus")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
