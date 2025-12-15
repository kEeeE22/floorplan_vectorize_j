from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os
import yaml
import uuid
import json

# Import các module cũ
from pp.pp_ import cleaning_img
from predict_segment.pred.predict_segment import predict_segment
from vectorize.vectorize import vectorize
from vectorize.utils import to_json

app = FastAPI(title="Floorplan Vectorization API")

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

MODEL_PATH = config['main']['model_path']

TEMP_DIR = "./output/temp_uploads"
DEFAULT_JSON_OUTPUT_DIR = os.path.join("output", "json_outputs") 

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(DEFAULT_JSON_OUTPUT_DIR, exist_ok=True)

@app.post("/vectorize")
async def vectorize_endpoint(file: UploadFile = File(...)):
    unique_id = str(uuid.uuid4())
    
    input_filename = f"{unique_id}_{file.filename}"
    input_path = os.path.join(TEMP_DIR, input_filename)

    output_json_name = f"{unique_id}_floorplan.json"
    
    real_output_path = os.path.join(DEFAULT_JSON_OUTPUT_DIR, output_json_name)

    generated_files = [input_path, real_output_path]

    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        room_path, boundary_path = predict_segment(input_path, MODEL_PATH, postprocess=True)
        
        if room_path: generated_files.append(room_path)
        if boundary_path: generated_files.append(boundary_path)

        clean_path = cleaning_img(boundary_path, room_path)
        if clean_path and clean_path not in generated_files: 
            generated_files.append(clean_path)

        # 3. Vector hóa
        optimized_graph = vectorize(input_path, clean_path)

        # 4. TẠO JSON OUTPUT
        # QUAN TRỌNG: Chỉ truyền tên file output_json_name
        print(f"Calling to_json with filename: {output_json_name}")
        to_json(optimized_graph.edges(data=True), file_name=output_json_name)

        # 5. Đọc file JSON lên để trả về
        # Lúc này ta mới dùng đường dẫn đầy đủ real_output_path để tìm file
        if os.path.exists(real_output_path):
            with open(real_output_path, 'r', encoding='utf-8') as jf:
                result_data = json.load(jf)
        else:
            # Fallback: Đôi khi to_json có thể ghi vào chỗ khác tùy version, kiểm tra lại
            print(f"Không tìm thấy file tại: {real_output_path}")
            raise FileNotFoundError(f"Output file not found at {real_output_path}")

        return result_data

    except Exception as e:
        print(f"Lỗi Server: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 6. Dọn dẹp
        for path in generated_files:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

if __name__ == "__main__":
    import uvicorn
    # Lưu ý: Truy cập bằng localhost thay vì 0.0.0.0 trên trình duyệt
    uvicorn.run(app, host="0.0.0.0", port=8000)