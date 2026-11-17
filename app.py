"""Interface graphique : QAOA pour MaxCut, avec initialisation apprise des angles.

Lancement local :  streamlit run app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from qaoaml import (  # noqa: E402
    LABELS, InitPredictor, QAOASimulator, brute_force, build_dataset, local_search, optimize,
    ramp_init, random_graph, random_init,
)

st.set_page_config(page_title="QAOA + IA", page_icon="⚛️", layout="wide")
SHORT = {"random": "Aléatoire", "ramp": "Rampe", "mean": "Moyenne", "knn": "k-NN", "rf": "Forêt", "mlp": "Réseau"}
COLORS = {"random": "#c9ced6", "ramp": "#8d99ae", "mean": "#4a5568", "knn": "#b9a3d8", "rf": "#9b7fc1",
          "mlp": "#6a4c93"}


@st.cache_resource(show_spinner="Préparation des modèles d'initialisation (une seule fois)…")
def predictors(p: int) -> dict:
    path = ROOT / "data" / f"train_p{p}.csv"
    df = pd.read_csv(path) if path.exists() else build_dataset(n_graphs=120, p=p, seed=0, workers=1)
    return {k: InitPredictor(k).fit(df) for k in ("mean", "knn", "rf", "mlp")}


def make_graph(kind: str, n: int, param: float, seed: int) -> nx.Graph:
    rng = np.random.default_rng(seed)
    if kind == "Erdős–Rényi":
        while True:
            g = nx.gnp_random_graph(n, param, seed=int(rng.integers(1 << 31)))
            if g.number_of_edges() >= 2:
                return g
    if kind == "Régulier":
        return nx.random_regular_graph(int(param), n, seed=int(rng.integers(1 << 31)))
    if kind == "Cycle":
        return nx.cycle_graph(n)
    return random_graph(rng, n, n)


# ------------------------------------------------------------------------------ barre latérale
st.sidebar.title("⚛️ Paramètres")
kind = st.sidebar.selectbox("Type de graphe", ["Aléatoire (mélange)", "Erdős–Rényi", "Régulier", "Cycle"])
n = st.sidebar.slider("Nombre de sommets (= qubits)", 5, 12, 8)
param = None
if kind == "Erdős–Rényi":
    param = st.sidebar.slider("Probabilité d'une arête", 0.2, 0.9, 0.5, 0.05)
elif kind == "Régulier":
    degrees = [d for d in (3, 4, 5) if (n * d) % 2 == 0 and d < n]
    param = st.sidebar.selectbox("Degré", degrees)
seed = st.sidebar.number_input("Graine du graphe", 0, 9999, 3)
p = st.sidebar.slider("Profondeur QAOA p", 1, 4, 2)
strategies = st.sidebar.multiselect("Initialisations à comparer", list(LABELS),
                                    default=["random", "ramp", "mean", "knn"], format_func=LABELS.get)
run = st.sidebar.button("🚀 Lancer QAOA", type="primary", width="stretch")
st.sidebar.caption(f"Simulation exacte de {n} qubits : {2 ** n} amplitudes.")

g = make_graph(kind, n, param, seed)
pos = nx.circular_layout(g)

st.title("QAOA pour MaxCut : l'IA choisit le point de départ")
st.markdown(
    "Le **QAOA** cherche une coupe maximale d'un graphe avec un circuit quantique à $2p$ angles réglables. "
    "Le choix des angles de départ décide du temps de calcul et de la qualité du résultat. "
    "Comparez des départs **au hasard**, **heuristiques** et **prédits par un modèle d'apprentissage**."
)

if run and strategies:
    sim = QAOASimulator(g)
    preds = predictors(p) if set(strategies) & {"mean", "knn", "rf", "mlp"} else {}
    rows, runs = [], {}
    rng = np.random.default_rng(0)
    for k in strategies:
        x0 = random_init(p, rng) if k == "random" else ramp_init(p) if k == "ramp" else preds[k].predict_params(g)
        t0 = time.perf_counter()
        res = optimize(sim, x0)
        runs[k] = (x0, res)
        rows.append({"Initialisation": LABELS[k], "Ratio au départ": sim.approx_ratio(x0),
                     "Ratio final": res.approx_ratio, "Évaluations du circuit": res.nfev,
                     "Temps (s)": time.perf_counter() - t0})
    t0 = time.perf_counter(); c_max, best_bits = brute_force(g); t_bf = time.perf_counter() - t0
    t0 = time.perf_counter(); c_ls, _ = local_search(g); t_ls = time.perf_counter() - t0
    st.session_state.r = dict(key=(kind, n, param, seed, p), sim=sim, runs=runs, table=pd.DataFrame(rows),
                              c_max=c_max, best_bits=best_bits, c_ls=c_ls, t_bf=t_bf, t_ls=t_ls)

r = st.session_state.get("r")
tab_res, tab_land, tab_learn = st.tabs(["📊 Résultats", "🗺️ Paysage (p = 1)", "📘 Comprendre"])

with tab_learn:
    st.markdown(r"""
### MaxCut
Séparer les sommets en deux groupes pour couper le plus d'arêtes possible. NP-difficile en général.

### QAOA
On prépare $|\gamma,\beta\rangle = U_M(\beta_p)U_C(\gamma_p)\cdots U_M(\beta_1)U_C(\gamma_1)|+\rangle^{\otimes n}$ :
$U_C$ donne à chaque partition une phase proportionnelle à sa coupe, $U_M$ les fait interférer.
Un optimiseur classique règle les $2p$ angles pour maximiser la coupe moyenne mesurée.
Le **ratio d'approximation** = coupe moyenne / coupe maximale.

### Pourquoi apprendre l'initialisation ?
Sur un vrai ordinateur quantique, chaque évaluation du circuit demande des centaines de mesures bruitées.
Or les angles optimaux **se ressemblent d'un graphe à l'autre** (concentration des paramètres) :
un modèle entraîné sur des petits graphes déjà résolus propose directement de bons angles.

### Les initialisations
- **Aléatoire** : la référence naïve.
- **Rampe linéaire** : heuristique inspirée du calcul adiabatique, sans apprentissage.
- **Moyenne** : moyenne des angles optimaux appris — la référence que le ML doit battre.
- **ML** : k plus proches voisins, forêt aléatoire, réseau de neurones, à partir de 12 caractéristiques du graphe.

### Et le classique ?
Sur ces petits graphes, la recherche exhaustive et la recherche locale trouvent l'optimum en quelques millisecondes.
L'intérêt du QAOA se jouerait sur de très grands graphes et du matériel quantique fiable — question de recherche ouverte.
""")

if r is None:
    with tab_res:
        c1, c2 = st.columns([2, 3])
        fig, ax = plt.subplots(figsize=(4, 4))
        nx.draw(g, pos, ax=ax, node_color="#c9ced6", with_labels=True)
        c1.pyplot(fig)
        c2.info("Réglez le graphe et les initialisations, puis cliquez sur **🚀 Lancer QAOA**.")
    with tab_land:
        st.info("Lancez un calcul avec p = 1 pour afficher le paysage.")
    st.stop()

if r["key"] != (kind, n, param, seed, p):
    st.caption("⚠️ Les paramètres ont changé depuis le dernier calcul : cliquez sur **Lancer** pour mettre à jour.")

sim, runs, table = r["sim"], r["runs"], r["table"]

with tab_res:
    best_k = max(runs, key=lambda k: (runs[k][1].approx_ratio, -runs[k][1].nfev))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Coupe maximale (exacte)", f"{r['c_max']} / {g.number_of_edges()} arêtes")
    m2.metric("Meilleur ratio QAOA", f"{runs[best_k][1].approx_ratio:.3f}")
    m3.metric(f"Recherche locale classique ({r['t_ls'] * 1e3:.1f} ms)", f"{r['c_ls'] / r['c_max']:.3f}")
    fewest = min(runs, key=lambda k: runs[k][1].nfev)
    m4.metric(f"Le moins d'évaluations : {SHORT[fewest]}", f"{runs[fewest][1].nfev} appels")

    st.dataframe(table.style.format({"Ratio au départ": "{:.3f}", "Ratio final": "{:.3f}", "Temps (s)": "{:.3f}"}),
                 width="stretch", hide_index=True)

    c1, c2, c3 = st.columns(3)
    fig, ax = plt.subplots(figsize=(4, 3.4))
    for k, (_, res) in runs.items():
        ax.plot(np.maximum.accumulate(res.history), color=COLORS[k], lw=1.6, label=SHORT[k])
    ax.set_xlabel("évaluations du circuit"); ax.set_ylabel("meilleur ratio atteint")
    ax.set_title("Convergence"); ax.legend(fontsize=7); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); c1.pyplot(fig)

    params = runs[best_k][1].params
    top = sim.most_likely(params, k=8)
    fig, ax = plt.subplots(figsize=(4, 3.4))
    labels = ["".join(map(str, b)) for b, _, _ in top]
    ax.barh(labels, [pr for _, pr, _ in top],
            color=["#d1495b" if c == r["c_max"] else "#8d99ae" for _, _, c in top])
    ax.invert_yaxis(); ax.set_xlabel("probabilité de mesure")
    ax.set_title("Partitions les plus probables\n(rouge = coupe optimale)", fontsize=9)
    ax.tick_params(axis="y", labelsize=7); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); c2.pyplot(fig)

    bits = top[0][0]
    fig, ax = plt.subplots(figsize=(4, 3.4))
    cut = [(i, j) for i, j in g.edges() if bits[i] != bits[j]]
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=["#6a4c93" if b else "#c9ced6" for b in bits])
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=8)
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=cut, edge_color="#d1495b", width=2)
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=[e for e in g.edges() if e not in cut], style="dashed")
    ax.set_title(f"Partition la plus probable : coupe {top[0][2]} / {r['c_max']}", fontsize=9); ax.axis("off")
    fig.tight_layout(); c3.pyplot(fig)

with tab_land:
    if p != 1:
        st.info("Le paysage complet n'est dessinable qu'avec 2 angles : choisissez **p = 1** et relancez.")
    else:
        G, B = np.meshgrid(np.linspace(-np.pi / 2, np.pi / 2, 70), np.linspace(-np.pi / 4, np.pi / 4, 70))
        Z = np.vectorize(lambda a, b: sim.approx_ratio([a, b]))(G, B)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        cs = ax.contourf(G, B, Z, levels=30, cmap="viridis")
        fig.colorbar(cs, label="ratio d'approximation")
        for k, (x0, res) in runs.items():
            x0w = (np.asarray(x0) + [np.pi / 2, np.pi / 4]) % [np.pi, np.pi / 2] - [np.pi / 2, np.pi / 4]
            xf = (res.params + [np.pi / 2, np.pi / 4]) % [np.pi, np.pi / 2] - [np.pi / 2, np.pi / 4]
            ax.annotate("", xf, x0w, arrowprops=dict(arrowstyle="->", color=COLORS[k], lw=1.5))
            ax.plot(*x0w, "o", color=COLORS[k], mec="white", ms=7, label=LABELS[k].replace("ML : ", ""))
        ax.set_xlabel(r"$\gamma$"); ax.set_ylabel(r"$\beta$")
        ax.set_title("Paysage QAOA p = 1 : départ (●) → arrivée de chaque initialisation")
        ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout(); st.pyplot(fig)
