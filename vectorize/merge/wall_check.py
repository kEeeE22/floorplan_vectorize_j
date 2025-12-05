import math
import numpy as np


def polyline_to_segments(polyline):
    segments = []
    for i in range(len(polyline) - 1):
        segments.append((tuple(polyline[i]), tuple(polyline[i+1])))
    return segments

def subdivide_segment(p1, p2, max_len=10):
    x1, y1 = p1
    x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length <= max_len:
        return [(p1, p2)]

    n_splits = int(np.ceil(length / max_len))
    xs = np.linspace(x1, x2, n_splits + 1)
    ys = np.linspace(y1, y2, n_splits + 1)

    sub_segments = []
    for i in range(n_splits):
        # Lưu ý: Ép kiểu int để tọa độ khớp nhau tuyệt đối
        sub_segments.append(((int(xs[i]), int(ys[i])), (int(xs[i+1]), int(ys[i+1]))))
    return sub_segments

def coverage_with_tolerance(p1, p2, seg_mask, tolerance=2, num_samples=5):
    x1, y1 = p1
    x2, y2 = p2
    h, w = seg_mask.shape
    xs = np.linspace(x1, x2, num_samples).astype(int)
    ys = np.linspace(y1, y2, num_samples).astype(int)
    valid_idx = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs = xs[valid_idx]
    ys = ys[valid_idx]
    if len(xs) == 0: return 0.0

    hit_count = 0
    for x, y in zip(xs, ys):
        y_min, y_max = max(0, y - tolerance), min(h, y + tolerance + 1)
        x_min, x_max = max(0, x - tolerance), min(w, x + tolerance + 1)
        if np.any(seg_mask[y_min:y_max, x_min:x_max] > 0):
            hit_count += 1
    return hit_count / len(xs)

def merge_consecutive_segments(segment_list):
    """
    Nối các đoạn thẳng nhỏ liên tiếp nhau thành 1 đoạn dài.
    Ví dụ: [(A,B), (B,C), (C,D)] -> [(A,D)]
    Chỉ nối khi điểm cuối đoạn trước == điểm đầu đoạn sau.
    """
    if not segment_list:
        return []

    merged = []
    # Khởi tạo đoạn đang xét bằng đoạn đầu tiên
    current_start, current_end = segment_list[0]

    for i in range(1, len(segment_list)):
        next_start, next_end = segment_list[i]

        # Nếu điểm cuối đoạn này trùng điểm đầu đoạn kia -> Nối
        if current_end == next_start:
            # Cập nhật điểm cuối mới, điểm đầu giữ nguyên
            current_end = next_end
        else:
            # Nếu bị ngắt quãng (do cửa sổ đã bị lọc mất), lưu đoạn hiện tại lại
            merged.append((current_start, current_end))
            # Bắt đầu đoạn mới
            current_start, current_end = segment_list[i]

    # Đừng quên lưu đoạn cuối cùng
    merged.append((current_start, current_end))
    return merged

def wall_line_check(raw_vectors, clean_wall_mask, threshold = 0.2, search_tolerance = 10, num_samples=5):
    filtered_segments = []

    for poly in raw_vectors:
        raw_segments = polyline_to_segments(poly)

        for (p_start, p_end) in raw_segments:
            # 1. Băm nhỏ
            sub_segs = subdivide_segment(p_start, p_end, max_len=10)

            # 2. Lọc (Giữ lại list các đoạn nhỏ hợp lệ)
            kept_sub_segs = []
            for (s1, s2) in sub_segs:
                ratio = coverage_with_tolerance(s1, s2, clean_wall_mask, tolerance=search_tolerance, num_samples=num_samples)
                if ratio >= threshold:
                    kept_sub_segs.append((s1, s2))

            # 3. HỢP NHẤT NGAY LẬP TỨC
            # Gom các đoạn nhỏ vừa lọc được lại thành đoạn dài
            merged_parts = merge_consecutive_segments(kept_sub_segs)

            # 4. Lưu vào kết quả cuối cùng
            filtered_segments.extend(merged_parts)
    return filtered_segments