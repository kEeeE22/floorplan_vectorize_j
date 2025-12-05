import pulp
from scipy.spatial import cKDTree
import numpy as np
import networkx as nx   

def smart_snap_optimization(G, snap_thresh=15.0):
    """
    Sử dụng Linear Programming (Pulp) để nối các điểm hở một cách thông minh,
    giữ nguyên hình dáng tường tốt nhất có thể.
    """
    print(f"[INFO] Bắt đầu tối ưu hóa vị trí nút (Threshold={snap_thresh})...")

    # 1. Lấy danh sách nút và tạo mapping index
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    coords = np.array(nodes)
    N = len(nodes)

    # 2. Dùng KDTree tìm các cặp điểm cần nối (Lấy từ code của bạn)
    tree = cKDTree(coords)
    # Tìm các cặp điểm cách nhau < snap_thresh
    pairs = tree.query_pairs(r=snap_thresh)

    if not pairs:
        print("Không tìm thấy điểm nào cần nối.")
        return G

    # 3. Thiết lập bài toán tối ưu (Pulp)
    prob = pulp.LpProblem("Smart_Snap", pulp.LpMinimize)

    # Biến số: Tọa độ mới (dx, dy) cho từng nút
    # gx, gy là toạ độ MỚI sau khi chỉnh sửa
    gx = {i: pulp.LpVariable(f"gx_{i}") for i in range(N)}
    gy = {i: pulp.LpVariable(f"gy_{i}") for i in range(N)}

    objective_terms = []

    # --- RÀNG BUỘC 1: HÀN GẮN (STITCHING) ---
    # Ép các cặp điểm tìm được phải trùng nhau
    for (i, j) in pairs:
        # Khoảng cách giữa điểm i và j sau khi dịch chuyển phải tiến về 0
        # Dùng biến phụ gap_x, gap_y để tính trị tuyệt đối
        gap_x = pulp.LpVariable(f"gap_x_{i}_{j}", 0)
        gap_y = pulp.LpVariable(f"gap_y_{i}_{j}", 0)

        prob += gx[i] - gx[j] <= gap_x
        prob += gx[j] - gx[i] <= gap_x
        prob += gy[i] - gy[j] <= gap_y
        prob += gy[j] - gy[i] <= gap_y

        # Phạt nặng nếu không trùng nhau (Priority cao nhất: 1000)
        objective_terms.append(1000 * (gap_x + gap_y))

    # --- RÀNG BUỘC 2: ỔN ĐỊNH VỊ TRÍ (STABILITY) ---
    # Không được chạy quá xa vị trí gốc (tránh hình bị biến dạng nát bét)
    for i in range(N):
        orig_x, orig_y = coords[i]

        dev_x = pulp.LpVariable(f"dev_x_{i}", 0)
        dev_y = pulp.LpVariable(f"dev_y_{i}", 0)

        prob += gx[i] - orig_x <= dev_x
        prob += orig_x - gx[i] <= dev_x
        prob += gy[i] - orig_y <= dev_y
        prob += orig_y - gy[i] <= dev_y

        # Priority thấp hơn: 1
        objective_terms.append(1 * (dev_x + dev_y))

    # 4. Giải bài toán
    prob += pulp.lpSum(objective_terms)
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if status != 1:
        print("Không tìm được giải pháp tối ưu!")
        return G

    # 5. Cập nhật lại Graph
    mapping = {}

    for i in range(N):
        old_node = nodes[i]
        new_x = int(round(gx[i].value()))
        new_y = int(round(gy[i].value()))
        mapping[old_node] = (new_x, new_y)

    # Relabel để gộp các nút trùng nhau
    new_G = nx.relabel_nodes(G, mapping, copy=True)

    print(f"Đã tối ưu xong. Số nút giảm từ {len(G.nodes())} xuống {len(new_G.nodes())}")
    return new_G