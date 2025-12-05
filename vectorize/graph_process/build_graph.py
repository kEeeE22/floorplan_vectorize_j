import math 
import networkx as nx
import numpy as np

from .utils import project_point_on_segment

def build_labeled_graph(walls, windows):
    G = nx.Graph()

    def add_polylines(polylines, label_type):
        for poly in polylines:
            for i in range(len(poly) - 1):
                p1 = poly[i]
                
                p2 = poly[i+1]

                # Làm tròn và chuyển về tuple để nối khớp các điểm gần nhau
                u = tuple(map(lambda x: int(round(x)), p1))
                v = tuple(map(lambda x: int(round(x)), p2))

                if u != v:
                    dist = math.hypot(u[0]-v[0], u[1]-v[1])
                    # QUAN TRỌNG: Gán thuộc tính 'kind' (loại) cho cạnh
                    G.add_edge(u, v, weight=dist, kind=label_type)

    # Thêm tường với nhãn 'wall'
    add_polylines(walls, label_type='wall')

    # Thêm cửa với nhãn 'window'
    add_polylines(windows, label_type='window')

    return G

def split_edges_by_nodes(G, tolerance=5.0):

    print("Đang xử lý cắt cạnh (Edge Splitting)...")

    # Lấy tọa độ tất cả các node
    nodes_pos = {n: np.array(n) for n in G.nodes()}
    nodes_list = list(G.nodes())

    # Danh sách các cạnh cần xóa và cần thêm
    edges_to_remove = []
    edges_to_add = []

    # Duyệt qua từng cạnh của đồ thị
    # Lưu ý: Convert sang list để tránh lỗi runtime khi graph thay đổi
    for u, v in list(G.edges()):
        p_u = nodes_pos[u]
        p_v = nodes_pos[v]

        # Danh sách các điểm nằm trên cạnh này (kể cả 2 đầu mút)
        points_on_edge = [(0.0, u), (1.0, v)] # (tỷ lệ t, node_id)

        # Duyệt qua tất cả các node khác trong graph để xem có ai nằm trên cạnh u-v không
        # (Cách này O(E*V), hơi chậm nếu graph quá lớn, nhưng ổn với floorplan)
        for node in nodes_list:
            if node == u or node == v: continue

            p_node = nodes_pos[node]

            # Kiểm tra khoảng cách và vị trí
            dist, t = project_point_on_segment(p_node, p_u, p_v)

            # Điều kiện: Nằm rất gần đường thẳng VÀ nằm giữa 2 đầu mút (0 < t < 1)
            # t > 0.01 và t < 0.99 để tránh trùng lặp 2 đầu mút
            if dist < tolerance and 0.01 < t < 0.99:
                points_on_edge.append((t, node))

        # Nếu tìm thấy điểm nào nằm giữa (tức là list > 2 phần tử ban đầu)
        if len(points_on_edge) > 2:
            edges_to_remove.append((u, v))

            # Sắp xếp các điểm theo thứ tự từ u đến v (dựa vào t)
            points_on_edge.sort(key=lambda x: x[0])

            # Tạo các cạnh nối tiếp: p1->p2, p2->p3...
            # Giữ lại thuộc tính (kind='wall'/'window') của cạnh gốc
            original_data = G.get_edge_data(u, v)

            for i in range(len(points_on_edge) - 1):
                n1 = points_on_edge[i][1]
                n2 = points_on_edge[i+1][1]

                # Tính lại weight (độ dài thật)
                dist_segment = np.linalg.norm(nodes_pos[n1] - nodes_pos[n2])

                # Copy thuộc tính từ cạnh mẹ
                attr = original_data.copy()
                attr['weight'] = dist_segment

                edges_to_add.append((n1, n2, attr))

    # Thực hiện cập nhật Graph
    G.remove_edges_from(edges_to_remove)
    for n1, n2, attr in edges_to_add:
        G.add_edge(n1, n2, **attr)

    print(f"Đã cắt {len(edges_to_remove)} cạnh và thêm mới {len(edges_to_add)} đoạn nhỏ.")
    return G

def zip_thick_walls(G, thickness_threshold=15):
    """
    Thay thế collapse_bubbles.
    Tìm các cạnh ngắn (ngang với độ dày tường) và gộp 2 đầu mút lại.
    Giúp "kéo khóa" các vòng tròn dẹt thành đường thẳng.
    """
    print("Đang thực hiện Zipping (Kéo khóa tường dày)...")

    # Lặp lại cho đến khi không còn cạnh ngắn nào để gộp
    while True:
        # Tìm các cạnh ngắn hơn độ dày tường (nhưng > 0)
        short_edges = []
        for u, v, data in G.edges(data=True):
            w = data.get('weight', 0)
            # Chỉ gộp nếu cạnh ngắn và không phải là cạnh nối chính nó
            if 0 < w < thickness_threshold and u != v:
                short_edges.append((u, v))

        if not short_edges:
            break

        # Lấy cạnh ngắn nhất để gộp trước
        u, v = short_edges[0]

        # Gộp v vào u (Contract edge)
        # NetworkX sẽ tự động chuyển các cạnh nối với v sang u
        G = nx.contracted_edge(G, (u, v), self_loops=False)

    return G

def simplify_graph_topology(G, collinear_threshold=0.1):
    """
    Phiên bản Fix: Chỉ xóa điểm thừa nếu 2 cạnh cùng loại (Wall-Wall hoặc Window-Window).
    Giữ lại điểm chuyển tiếp giữa Tường và Cửa sổ.
    """
    print("Đang làm sạch các điểm thừa (Giữ lại điểm tiếp giáp)...")

    while True:
        nodes_to_remove = []
        processed_nodes = set()

        for node in G.nodes():
            if G.degree(node) == 2:
                neighbors = list(G.neighbors(node))
                u, v = neighbors[0], neighbors[1]

                # --- FIX 1: KIỂM TRA LOẠI CẠNH ---
                # Lấy loại của cạnh 1 và cạnh 2
                kind1 = G.get_edge_data(u, node).get('kind')
                kind2 = G.get_edge_data(node, v).get('kind')

                # Nếu khác loại (1 cái Tường, 1 cái Cửa) -> TUYỆT ĐỐI KHÔNG XÓA
                if kind1 != kind2:
                    continue
                # ---------------------------------

                # Kiểm tra thẳng hàng (Logic cũ)
                p_u, p_n, p_v = np.array(u), np.array(node), np.array(v)
                vec1, vec2 = p_n - p_u, p_v - p_n
                norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)

                if norm1 == 0 or norm2 == 0: continue

                cross = (vec1[0]*vec2[1] - vec1[1]*vec2[0]) / (norm1 * norm2)

                if abs(cross) < collinear_threshold:
                    if u not in processed_nodes and v not in processed_nodes:
                        nodes_to_remove.append((node, u, v))
                        processed_nodes.add(node)

        if not nodes_to_remove:
            break

        count_removed = 0
        for node, u, v in nodes_to_remove:
            if not G.has_node(node): continue
            if not G.has_edge(u, node) or not G.has_edge(node, v): continue

            # Copy thuộc tính (lúc này đã an toàn vì kind1 == kind2)
            attr = G.get_edge_data(u, node).copy()
            new_weight = np.linalg.norm(np.array(u) - np.array(v))
            attr['weight'] = new_weight

            G.add_edge(u, v, **attr)
            G.remove_node(node)
            count_removed += 1

        if count_removed == 0:
            break

    print(f"Đã làm sạch. Số nút hiện tại: {G.number_of_nodes()}")
    return G

def orthogonalize_graph(G, angle_threshold=15, iterations=10):
    """
    Nắn thẳng toàn bộ đồ thị bằng cách lặp đi lặp lại việc căn chỉnh toạ độ.

    Args:
        G: Graph đầu vào.
        angle_threshold: Góc lệch tối đa (độ) cho phép nắn về 0 hoặc 90.
                         (Ví dụ: 15 độ nghĩa là nghiêng < 15 độ sẽ bị bẻ thẳng).
        iterations: Số lần lặp để lan truyền sự thẳng hàng (càng nhiều càng thẳng).
    """
    print(f"Đang cưỡng bức nắn thẳng (Force Orthogonality) - {iterations} vòng...")

    pos = {n: [float(n[0]), float(n[1])] for n in G.nodes()}
    nodes = list(G.nodes())

    rad_thresh = math.radians(angle_threshold)

    for _ in range(iterations):
        moves_x = {n: [] for n in nodes}
        moves_y = {n: [] for n in nodes}

        for u, v in G.edges():
            p_u = pos[u]
            p_v = pos[v]

            dx = p_v[0] - p_u[0]
            dy = p_v[1] - p_u[1]

            angle = math.atan2(dy, dx)

            if abs(dy) < abs(dx) * math.tan(rad_thresh):
                avg_y = (p_u[1] + p_v[1]) / 2.0
                moves_y[u].append(avg_y)
                moves_y[v].append(avg_y)

            elif abs(dx) < abs(dy) * math.tan(rad_thresh):
                avg_x = (p_u[0] + p_v[0]) / 2.0
                moves_x[u].append(avg_x)
                moves_x[v].append(avg_x)

        for n in nodes:
            if moves_x[n]:
                pos[n][0] = sum(moves_x[n]) / len(moves_x[n])

            if moves_y[n]:
                pos[n][1] = sum(moves_y[n]) / len(moves_y[n])

    mapping = {}
    for n in nodes:
        new_x = int(round(pos[n][0]))
        new_y = int(round(pos[n][1]))
        mapping[n] = (new_x, new_y)

    G_final = nx.relabel_nodes(G, mapping, copy=True)

    G_final.remove_edges_from(nx.selfloop_edges(G_final))

    return G_final