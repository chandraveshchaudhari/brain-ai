from typing import Optional

try:
    import matplotlib.pyplot as plt
    import networkx as nx
except Exception:  # pragma: no cover - optional
    plt = None
    nx = None


def save_graph_png(G, path: str):
    if plt is None or nx is None:
        raise RuntimeError("matplotlib or networkx not available")
    plt.figure(figsize=(8, 4))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=1200, node_color="#A6CEE3")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
