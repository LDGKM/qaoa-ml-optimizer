"""qaoa-ml-optimizer : QAOA pour MaxCut, avec initialisation des angles apprise."""

from .graphs import FEATURES, graph_features, random_graph
from .maxcut import brute_force, cut_value, local_search
from .qaoa import QAOASimulator, canonicalize, optimize, pennylane_circuit
from .init import InitPredictor, ramp_init, random_init
from .dataset import build_dataset, graph_from_row

__version__ = "0.1.0"
from .benchmark import STRATEGIES, LABELS, run_benchmark, evaluate, summarize
