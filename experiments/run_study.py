"""Étude expérimentale complète (résultats du rapport docs/rapport.tex).

  A. profondeur p = 1, 2, 3 : toutes les stratégies (200 graphes d'entraînement, 50 de test) ;
  B. courbe d'apprentissage (p = 2) : combien de graphes faut-il pour bien prédire ?
  C. généralisation (p = 2) : entraîné sur 6 à 10 sommets, testé sur 12 sommets.

    python experiments/run_study.py      # ~5 min sur 2 cœurs (datasets mis en cache dans results/)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from qaoaml.benchmark import LABELS, STRATEGIES, evaluate, load_or_build, summarize  # noqa: E402

RES = ROOT / "results"
OUT = RES / "study"
SHORT = {"random": "Aléatoire", "ramp": "Rampe", "mean": "Moyenne", "knn": "ML k-NN",
         "rf": "ML forêt", "mlp": "ML MLP"}


def data(p, n, n_min=6, n_max=10, seed=0, kind="train"):
    return load_or_build(RES / f"{kind}_p{p}_n{n_min}-{n_max}_{n}.csv", n_graphs=n, p=p,
                         n_min=n_min, n_max=n_max, seed=seed)


def exp_depth():
    rows = []
    for p in (1, 2, 3):
        print(f"[A] p = {p}")
        res = evaluate(data(p, 200), data(p, 50, seed=1, kind="test"))
        s = summarize(res)
        s.insert(0, "p", p)
        s["key"] = [k for k in STRATEGIES]
        rows.append(s)
    return pd.concat(rows)


def exp_learning_curve():
    train, test = data(2, 200), data(2, 50, seed=1, kind="test")
    rows = []
    for n in (5, 10, 25, 50, 100, 200):
        print(f"[B] {n} graphes d'entraînement")
        res = evaluate(train.iloc[:n], test, strategies=["mean", "knn", "rf", "mlp"])
        for k, g in res.groupby("strategy"):
            rows.append({"n_train": n, "strategy": k, "init_ratio": g.init_ratio.mean(),
                         "init_ratio_std": g.init_ratio.std(), "nfev": g.nfev.mean()})
    return pd.DataFrame(rows)


def exp_generalization():
    print("[C] test sur des graphes de 12 sommets")
    big = data(2, 30, n_min=12, n_max=12, seed=2, kind="test")
    res = evaluate(data(2, 200), big)
    s = summarize(res)
    s["key"] = STRATEGIES
    return s


def table_depth(d: pd.DataFrame) -> str:
    lines = [r"\begin{tabular}{l" + "ccc" * 1 + "ccc}", r"\toprule",
             r" & \multicolumn{3}{c}{Ratio sans optimisation} & \multicolumn{3}{c}{Évaluations du circuit} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Stratégie & $p=1$ & $p=2$ & $p=3$ & $p=1$ & $p=2$ & $p=3$ \\", r"\midrule"]
    for k in STRATEGIES:
        r = d[d.key == k].set_index("p")
        best_i = d.groupby("p").init_ratio.max()
        best_n = d.groupby("p").nfev.min()
        ci = [(r"\textbf{%.3f}" if np.isclose(r.loc[p, "init_ratio"], best_i[p]) else "%.3f") % r.loc[p, "init_ratio"]
              for p in (1, 2, 3)]
        cn = [(r"\textbf{%.0f}" if np.isclose(r.loc[p, "nfev"], best_n[p]) else "%.0f") % r.loc[p, "nfev"]
              for p in (1, 2, 3)]
        if k == "knn":
            lines.append(r"\midrule")
        lines.append(f"{LABELS[k]} & " + " & ".join(ci + cn) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_final(d: pd.DataFrame) -> str:
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Stratégie & $p=1$ & $p=2$ & $p=3$ \\", r"\midrule"]
    for k in STRATEGIES:
        r = d[d.key == k].set_index("p")
        lines.append(f"{LABELS[k]} & " + " & ".join(
            f"{r.loc[p, 'final_ratio']:.3f} ({100 * r.loc[p, 'success']:.0f}\\,\\%)" for p in (1, 2, 3)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_gen(s: pd.DataFrame) -> str:
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Stratégie & Ratio initial & Ratio final & Évaluations \\", r"\midrule"]
    for _, r in s.iterrows():
        lines.append(f"{r['stratégie']} & {r.init_ratio:.3f} & {r.final_ratio:.3f} & {r.nfev:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def figures(depth: pd.DataFrame, lc: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    col = {"random": "#c9ced6", "ramp": "#8d99ae", "mean": "#4a5568", "knn": "#b9a3d8", "rf": "#9b7fc1",
           "mlp": "#6a4c93"}

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.3))
    for k in STRATEGIES:
        r = depth[depth.key == k]
        axes[0].plot(r.p, r.init_ratio, "o-", ms=3, lw=1.2, color=col[k], label=SHORT[k])
        axes[1].plot(r.p, r.nfev, "o-", ms=3, lw=1.2, color=col[k])
    axes[0].set_ylabel("Ratio sans optimisation"); axes[1].set_ylabel("Évaluations du circuit")
    for ax in axes:
        ax.set_xlabel("Profondeur $p$"); ax.set_xticks([1, 2, 3])
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=6, loc="upper center", frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "fig_depth.pdf"); fig.savefig(OUT / "fig_depth.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for k in ["mean", "knn", "rf", "mlp"]:
        r = lc[lc.strategy == k]
        ax.plot(r.n_train, r.init_ratio, "o-", ms=3, lw=1.2, color=col[k], label=SHORT[k])
    ax.set_xscale("log"); ax.set_xlabel("Graphes d'entraînement"); ax.set_ylabel("Ratio sans optimisation")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "fig_learning_curve.pdf"); fig.savefig(OUT / "fig_learning_curve.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    depth = exp_depth(); depth.to_csv(OUT / "depth.csv", index=False)
    lc = exp_learning_curve(); lc.to_csv(OUT / "learning_curve.csv", index=False)
    gen = exp_generalization(); gen.to_csv(OUT / "generalization.csv", index=False)
    (OUT / "table_depth.tex").write_text(table_depth(depth), encoding="utf-8")
    (OUT / "table_final.tex").write_text(table_final(depth), encoding="utf-8")
    (OUT / "table_generalization.tex").write_text(table_gen(gen), encoding="utf-8")
    figures(depth, lc)
    print(f"Résultats dans {OUT}")


if __name__ == "__main__":
    main()
