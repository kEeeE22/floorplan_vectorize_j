from pp.pp_ import cleaning_img
from predict_segment.pred.predict_segment import predict_segment
from vectorize.vectorize import vectorize
from vectorize.utils import plot_graph, to_json
import os
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

if __name__ == "__main__":

    image_path = config['main']['img_path']
    loadmodel = config['main']['model_path']
    plotting = config['main']['plotting']

    print('Đang xử lý')
    print('='*50)
    room_path, boundary_path = predict_segment(image_path, loadmodel, postprocess=True)

    clean_path = cleaning_img(boundary_path, room_path)

    print('Vector hóa')
    optimized_graph = vectorize(image_path, clean_path)


    json_name = os.path.splitext(os.path.basename(image_path))[0] + '_floorplan_j.json'
    to_json(optimized_graph.edges(data=True), file_name=json_name)
    print('Lưu vào {}'.format(json_name))
    print('='*50)
    if plotting:
        plot_graph(optimized_graph)

    print('Xong')