import networkx as nx
import matplotlib.pyplot as plt

from .pulp_ import smart_snap_optimization
from .build_graph import orthogonalize_graph, zip_thick_walls, simplify_graph_topology, split_edges_by_nodes


class GraphOptimizer:
    def __init__(self, G):
        self.G = G

    def optimize(self):
        new_g = split_edges_by_nodes(self.G, tolerance=5)
        new_g = smart_snap_optimization(new_g, snap_thresh=10)

        new_g = zip_thick_walls(new_g, 5)

        new_g.remove_edges_from(nx.selfloop_edges(new_g))

        new_new_g = orthogonalize_graph(new_g, angle_threshold=45, iterations=30)
        new_new_g = simplify_graph_topology(new_new_g, collinear_threshold=0.2)

        return new_new_g
    
