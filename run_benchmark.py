"""Compare les stratégies d'initialisation du QAOA sur des graphes jamais vus.

Exemples :
    python run_benchmark.py                 # p = 2, 200 graphes d'entraînement, 50 de test
    python run_benchmark.py --p 3 --train 300
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from qaoaml import run_benchmark  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="QAOA + ML : initialisation apprise des angles")
    ap.add_argument("--p", type=int, default=2, help="profondeur du circuit QAOA")
    ap.add_argument("--train", type=int, default=200, help="graphes d'entraînement")
    ap.add_argument("--test", type=int, default=50, help="graphes de test")
    ap.add_argument("--nmin", type=int, default=6, help="sommets minimum")
    ap.add_argument("--nmax", type=int, default=10, help="sommets maximum (2^n états simulés)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    summary, classical, _ = run_benchmark(p=a.p, n_train=a.train, n_test=a.test,
                                          n_min=a.nmin, n_max=a.nmax, seed=a.seed)
    print()
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nRéférence classique — recherche locale : ratio moyen {classical.local_search_ratio.mean():.3f} "
          f"en {classical.t_local_search.mean() * 1e3:.1f} ms ; force brute : "
          f"{classical.t_bruteforce.mean() * 1e3:.1f} ms par graphe.")
    print("Résultats enregistrés dans results/")


if __name__ == "__main__":
    main()
