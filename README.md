# Floorplan Vectorize

Công cụ vector hóa mặt bằng tự động từ ảnh floorplan.

## Yêu cầu hệ thống
- Hiện tại đang chạy trên Python 3.13.4
- CUDA (khuyến nghị cho xử lý nhanh hơn)

## Cài đặt

1. Clone repository:
```bash
git clone https://github.com/kEeeE22/floorplan_vectorize_j.git
cd pp_export
```

2. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## Cấu hình

Chỉnh sửa file `config.yaml`:

```yaml
main:
  img_path: 'đường/dẫn/đến/ảnh/floorplan.jpg'  # Đường dẫn ảnh đầu vào
  model_path: 'predict_segment/checkpoint/tên checkpoint (file .pt)'  # Đường dẫn model
  plotting: 'False'  # True: hiển thị đồ thị, False: không hiển thị
```

## Cách sử dụng

### Chạy chương trình

```bash
python main.py
```

### Quy trình xử lý

Chương trình sẽ tự động thực hiện các bước:

1. **Phân đoạn ảnh (Segmentation)**: Phát hiện phòng và ranh giới tường
2. **Làm sạch ảnh (Cleaning)**: Xử lý và làm sạch kết quả phân đoạn
3. **Vector hóa (Vectorization)**: Chuyển đổi ảnh thành đồ thị vector
4. **Xuất kết quả**: Lưu file JSON và ảnh kết quả

### Kết quả đầu ra

Các file kết quả được lưu trong thư mục `output/`:

- `output/segment_output/`: Ảnh phân đoạn phòng và ranh giới
  - `*_post_room.png`: Ảnh phân đoạn phòng
  - `*_post_boundary.png`: Ảnh ranh giới tường
  
- `output/pp_output/`: Ảnh sau xử lý
  - `*_post_clean.png`: Ảnh đã làm sạch
  
- `output/json_outputs/`: File JSON chứa dữ liệu vector
  - `*_floorplan_j.json`: Dữ liệu đồ thị vector (nodes, edges)

## Cấu trúc thư mục

```
pp_export/
├── pp/                    # Module xử lý và làm sạch ảnh
├── predict_segment/       # Module phân đoạn ảnh
│   ├── checkpoint/       # Model đã train
│   └── pred/             # Code dự đoán
├── vectorize/            # Module vector hóa
│   ├── graph_process/    # Xử lý đồ thị
│   └── merge/            # Gộp và kiểm tra tường
├── output/               # Thư mục chứa kết quả
├── config.yaml           # File cấu hình
├── main.py               # File chạy chính
└── requirements.txt      # Danh sách thư viện
```

## Ví dụ

```bash
# 1. Cấu hình đường dẫn ảnh trong config.yaml
# 2. Chạy chương trình
python main.py

# Kết quả:
# Đang xử lý
# ==================================================
# Vector hóa
# Lưu vào 47499272_floorplan_j.json
# ==================================================
# Xong
```

## Lưu ý

- Ảnh đầu vào nên có độ phân giải tốt để kết quả chính xác
- Model checkpoint phải tồn tại tại đường dẫn đã cấu hình
- Đặt `plotting: True` để xem trực quan kết quả đồ thị (cần GUI)

## Hỗ trợ

Nếu gặp lỗi, kiểm tra:
- Đường dẫn ảnh và model trong `config.yaml` có đúng không
- Các thư viện đã được cài đặt đầy đủ chưa
- Định dạng ảnh đầu vào (hỗ trợ: jpg, png)
