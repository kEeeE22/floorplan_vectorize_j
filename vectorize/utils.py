import matplotlib.pyplot as plt
import networkx as nx
import json
import os

def plot_graph(new_new_g):
    plt.figure(figsize=(12, 12))
    pos = {n: n for n in new_new_g.nodes()}

    wall_edges = [(u, v) for u, v, d in new_new_g.edges(data=True) if d.get('kind') == 'wall']
    window_edges = [(u, v) for u, v, d in new_new_g.edges(data=True) if d.get('kind') == 'window']

    nx.draw_networkx_edges(new_new_g, pos, edgelist=wall_edges, edge_color='orange', width=3, label='Tường')
    nx.draw_networkx_edges(new_new_g, pos, edgelist=window_edges, edge_color='cyan', width=3, style='dashed', label='Cửa sổ')
    nx.draw_networkx_nodes(new_new_g, pos, node_size=30, node_color='red')

    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.legend()
    plt.show()

# def to_json(edges, default_thickness=150, file_name='floorplan.json'):
#     """
#     edges = list chứa tuples (u, v, info_dict)
#     u, v = (x, y)
#     info = {'kind': 'wall'/'window', 'weight': ...}
#     """

#     data = {
#         "units": "mm",
#         "vertices": {},
#         "walls": {},
#         "rooms": {},
#         "symbols": {},
#         "instances": {}
#     }

#     # -----------------------------------------------
#     # 1. Thu thập toàn bộ node thành danh sách unique
#     # -----------------------------------------------
#     nodes = {}
#     for u, v, d in edges:
#         nodes[u] = None
#         nodes[v] = None

#     # Gán ID v1, v2, v3...
#     node_map = {}
#     for i, node in enumerate(nodes.keys()):
#         node_map[node] = f"v{i+1}"
#         x, y = node
#         data["vertices"][node_map[node]] = {"x": float(x), "y": float(y)}

#     # -----------------------------------------------
#     # 2. Tạo walls và windows
#     # -----------------------------------------------
#     wall_id = 1
#     inst_id = 1

#     for (u, v, d) in edges:
#         uid = node_map[u]
#         vid = node_map[v]
#         kind = d.get("kind", "wall")

#         if kind == "wall":
#             wid = f"w{wall_id}"
#             data["walls"][wid] = {
#                 "vStart": uid,
#                 "vEnd": vid,
#                 "thickness": default_thickness,
#                 "isOuter": False
#             }
#             wall_id += 1

#         elif kind == "window":
#             iid = f"win{inst_id}"
#             offset = float(d.get("weight", 0)) / 2

#             data["instances"][iid] = {
#                 "symbol": "window.slider",
#                 "constraint": {
#                     "attachTo": {"kind": "wall", "id": "UNKNOWN"},
#                     "offsetFromStart": offset
#                 },
#                 "props": {"width": d.get("weight", 100)}
#             }
#             inst_id += 1

#     basepath = './output/json_outputs/'
#     os.makedirs(basepath, exist_ok=True)
#     filepath = os.path.join(basepath, file_name)
#     with open(filepath, 'w') as f:
#         json.dump(data, f, indent=2, ensure_ascii=False)

#v2
def to_json(edges, default_thickness=150, file_name='floorplan_minicad.json'):
    """
    edges = list chứa tuples (u, v, info_dict)
    info_dict có thể chứa 'is_window': True hoặc 'kind': 'window'
    """

    data = {
        "units": "mm",
        "vertices": {},
        "walls": {},
        "rooms": {},
        "symbols": {},
        "instances": {}
    }

    # -----------------------------------------------
    # 1. Thu thập toàn bộ node và tạo vertices
    # -----------------------------------------------
    nodes = {}
    for u, v, d in edges:
        nodes[u] = None
        nodes[v] = None

    node_map = {}
    for i, node in enumerate(nodes.keys()):
        # Tạo ID vertex v1, v2...
        v_id = f"v{i+1}"
        node_map[node] = v_id
        x, y = node
        data["vertices"][v_id] = {"x": float(x), "y": float(y)}

    # -----------------------------------------------
    # 2. Tạo walls (cho TẤT CẢ cạnh) và windows (nếu có)
    # -----------------------------------------------
    wall_counter = 1
    inst_counter = 1

    for (u, v, d) in edges:
        uid = node_map[u]
        vid = node_map[v]

        # --- BƯỚC A: LUÔN TẠO TƯỜNG ---
        current_wall_id = f"w{wall_counter}"

        data["walls"][current_wall_id] = {
            "vStart": uid,
            "vEnd": vid,
            "thickness": default_thickness,
            "isOuter": False
        }

        # --- BƯỚC B: NẾU LÀ CỬA SỔ -> TẠO INSTANCE ĐÈ LÊN ---
        # Kiểm tra cờ is_window (từ code trước) hoặc kind (từ code cũ)
        is_window = d.get("is_window") is True or d.get("kind") == "window"

        if is_window:
            iid = f"win{inst_counter}"
            length = float(d.get("weight", 0))

            # Đặt cửa sổ nằm giữa đoạn tường này
            offset = length / 2

            data["instances"][iid] = {
                "symbol": "window.slider",
                "constraint": {
                    # Quan trọng: Attach ngay vào cái tường (current_wall_id) vừa tạo ở trên
                    "attachTo": {"kind": "wall", "id": current_wall_id},
                    "offsetFromStart": offset
                },
                "props": {
                    "width": length # Chiều rộng cửa sổ bằng đúng chiều dài đoạn tường
                }
            }
            inst_counter += 1

        # Tăng wall_counter sau khi xử lý xong cặp (u,v) này
        wall_counter += 1
    basepath = './output/json_outputs/'
    os.makedirs(basepath, exist_ok=True)
    filepath = os.path.join(basepath, file_name)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)