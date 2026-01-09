import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def project_point_on_segment(p, a, b):
    """
    Tìm hình chiếu của điểm p lên đoạn thẳng ab.
    Trả về: khoảng cách, và tỷ lệ t (0 <= t <= 1 nghĩa là nằm giữa)
    """
    ap = p - a
    ab = b - a

    # Độ dài bình phương đoạn ab
    ab2 = np.dot(ab, ab)
    if ab2 == 0: return np.linalg.norm(ap), 0 # a trùng b

    # Tính tỷ lệ t hình chiếu
    t = np.dot(ap, ab) / ab2

    if t < 0.0: # Điểm chiếu nằm ngoài phía a
        dist = np.linalg.norm(p - a)
    elif t > 1.0: # Điểm chiếu nằm ngoài phía b
        dist = np.linalg.norm(p - b)
    else: # Điểm chiếu nằm trong đoạn ab
        projection = a + t * ab
        dist = np.linalg.norm(p - projection)

    return dist, t

def plot_graph(new_new_g):
        plt.figure(figsize=(12, 12))
        pos = {n: n for n in new_new_g.nodes()}

        wall_edges = [(u, v) for u, v, d in new_new_g.edges(data=True) if d.get('is_window') == False]
        window_edges = [(u, v) for u, v, d in new_new_g.edges(data=True) if d.get('is_window') == True]

        nx.draw_networkx_edges(new_new_g, pos, edgelist=wall_edges, edge_color='orange', width=3, label='Tường')
        nx.draw_networkx_edges(new_new_g, pos, edgelist=window_edges, edge_color='cyan', width=3, style='dashed', label='Cửa sổ')
        nx.draw_networkx_nodes(new_new_g, pos, node_size=30, node_color='red')

        plt.gca().invert_yaxis()
        plt.axis('equal')
        plt.legend()
        plt.show()