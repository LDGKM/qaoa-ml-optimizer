# qaoa-ml-optimizer

**Un modèle d'apprentissage peut-il dire à un algorithme quantique par où commencer ?**

Le **QAOA** (*Quantum Approximate Optimization Algorithm*) résout des problèmes d'optimisation comme
**MaxCut** avec un circuit quantique dont $2p$ angles sont réglés par un optimiseur classique.
Le choix des angles de départ détermine le nombre d'appels au circuit (le vrai coût sur un ordinateur quantique)
et le risque de rester bloqué dans un mauvais optimum.

Ce projet entraîne des modèles (k plus proches voisins, forêt aléatoire, réseau de neurones) à **prédire les angles
optimaux à partir de la structure du graphe**, et les compare à des départs aléatoires, à une rampe linéaire
inspirée du calcul adiabatique, et à la moyenne des angles connus.

Projet 2 sur 4 d'une série IA × informatique quantique :
[`quantum-vs-classical-classifier`](https://github.com/LDGKM/quantum-vs-classical-classifier) ·
[`quantum-circuit-agent`](https://github.com/LDGKM/quantum-circuit-agent) ·
[`neural-quantum-decoder`](https://github.com/LDGKM/neural-quantum-decoder)

---

## Résultats principaux

Graphes de 6 à 10 sommets, 200 graphes d'entraînement, 50 graphes de test jamais vus
(`python experiments/run_study.py`) :

| Initialisation ($p = 3$) | Ratio **sans** optimisation | Évaluations du circuit | Meilleur optimum trouvé |
|---|---|---|---|
| Aléatoire | 0,597 | 319 | 10 % des graphes |
| Rampe linéaire | 0,834 | 126 | 98 % |
| Moyenne des angles appris | 0,902 | 103 | 92 % |
| **ML : k plus proches voisins** | **0,921** | **70** | **98 %** |

- Le modèle propose des angles qui atteignent **99 % de l'optimum sans aucune optimisation**, et l'optimisation
  qui suit demande **44 % d'appels en moins** que la rampe et **4,5 fois moins** qu'un départ aléatoire.
- **5 graphes d'entraînement** suffisent déjà à battre la moyenne : les angles optimaux sont très « concentrés ».
- Entraîné sur 6 à 10 sommets, le modèle **généralise à 12 sommets** (99,8 % de l'optimum au départ).
- Honnêtement : sur ces petits graphes, une recherche locale classique trouve la coupe **exacte** en moins d'une
  milliseconde. L'enjeu est de rendre le QAOA moins coûteux sur du matériel quantique réel, pas de battre le classique ici.

## Utiliser l'application (rien à installer)

| | Comment |
|---|---|
| 🌐 **En ligne** | à déployer sur Streamlit Community Cloud (voir plus bas) |
| 🪟 🍎 🐧 **Hors ligne** | télécharger le fichier de son système dans les [Releases](https://github.com/LDGKM/qaoa-ml-optimizer/releases) |

Choisissez un graphe, la profondeur $p$ et les initialisations : l'application affiche le ratio obtenu, le nombre
d'appels au circuit, les courbes de convergence, les partitions les plus probables et, pour $p = 1$,
le **paysage d'énergie** avec la trajectoire de chaque initialisation.

### Mise en ligne (une seule fois)

**Application web** — sur [share.streamlit.io](https://share.streamlit.io) : *Create app* → dépôt
`LDGKM/qaoa-ml-optimizer`, branche `main`, **Main file path** `streamlit_cloud/streamlit_app.py`,
*Advanced settings* → Python 3.12 → *Deploy*.

**Exécutables** — déplacer le workflow à sa place puis créer un tag :
```powershell
New-Item -ItemType Directory -Force .github\workflows
Move-Item packaging\github-workflows\build-apps.yml .github\workflows\
git add . ; git commit -m "Workflow des exécutables" ; git push
git tag v0.1.0 ; git push origin v0.1.0
```

## Installation pour développer

Prérequis : Python 3.10 à 3.13. Windows : double-cliquer sur **`setup.bat`** ; Linux/macOS : `bash setup.sh`.
Le script crée `.venv`, installe `requirements.txt` et le noyau Jupyter **Python (qaoa-ml)**.

```bash
.venv\Scripts\activate             # Windows (Linux/macOS : source .venv/bin/activate)
python -m pytest                   # 6 tests, ~5 s
python run_benchmark.py            # p = 2, ~30 s
python run_benchmark.py --p 3
python experiments/run_study.py    # étude complète du rapport, ~5 min
streamlit run app.py               # interface graphique
jupyter lab                        # notebooks
```

## Comment ça marche

1. **Données** : 200 graphes aléatoires (Erdős–Rényi et réguliers, 6 à 10 sommets). Pour chacun, on cherche les
   meilleurs angles QAOA (départ en rampe + 3 départs aléatoires) et on calcule 12 caractéristiques
   (degrés, densité, triangles, spectre du laplacien, bipartition…).
2. **Canonisation** : les angles ont des symétries ($\gamma \to \gamma + 2\pi$, $\beta \to \beta + \pi/2$,
   $(\gamma,\beta) \to (-\gamma,-\beta)$). On les ramène dans une zone de référence, sinon deux solutions
   équivalentes auraient des coordonnées différentes et brouilleraient l'apprentissage.
3. **Apprentissage** : régression caractéristiques → $2p$ angles.
4. **Évaluation** : sur des graphes jamais vus, on optimise depuis chaque initialisation et on compte les appels.

La simulation utilise un **simulateur spécialisé** (`QAOASimulator`), environ 50 fois plus rapide que PennyLane,
dont l'exactitude est vérifiée par les tests contre le vrai circuit PennyLane (`pennylane_circuit`).

## Structure

```
├── app.py                      interface graphique (Streamlit)
├── run_benchmark.py            comparaison en ligne de commande
├── setup.bat / setup.sh        installation (développement)
├── requirements.txt            dépendances de développement
├── requirements-app.txt        dépendances de l'application seule
├── src/qaoaml/
│   ├── graphs.py               génération de graphes, 12 caractéristiques
│   ├── maxcut.py               coupe, force brute, recherche locale
│   ├── qaoa.py                 simulateur rapide, circuit PennyLane, optimisation, canonisation
│   ├── init.py                 stratégies d'initialisation et modèles d'apprentissage
│   ├── dataset.py              construction du jeu de données graphe → angles optimaux
│   └── benchmark.py            évaluation et export CSV / LaTeX / PNG
├── experiments/run_study.py    étude complète (p = 1 à 3, courbe d'apprentissage, généralisation)
├── notebooks/01_qaoa_maxcut.ipynb
├── tests/test_smoke.py
├── data/                       jeux d'entraînement utilisés par l'application (p = 1 à 4)
├── docs/rapport.tex            rapport (LaTeX)
├── results/                    résultats des benchmarks et de l'étude
├── streamlit_cloud/            point d'entrée de l'application en ligne
├── packaging/                  exécutables (PyInstaller) et workflow GitHub Actions
└── Dockerfile                  image de l'application (hébergement alternatif)
```

## Feuille de route

- [x] QAOA MaxCut, simulateur rapide vérifié contre PennyLane
- [x] Initialisation apprise, comparaison à 5 références, étude complète
- [x] Interface graphique, exécutables
- [ ] Graphes pondérés, plus grands graphes (simulateur à vecteur d'état limité à ~20 qubits)
- [ ] Effet du bruit et des mesures finies ; exécution sur IBM Quantum
- [ ] Prédire avec un réseau de neurones sur graphes (GNN) plutôt qu'avec des caractéristiques choisies à la main

## Licence

MIT
