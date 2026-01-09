import cv2
import os

from pp.utils import clean_opening_to_wall, close_window_edges
import yaml

with open('./config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def cleaning_img(boundary_path, room_path):
    boundary = cv2.imread(boundary_path)
    room = cv2.imread(room_path)

    img = cv2.add(boundary, room)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    clean_img1 = clean_opening_to_wall(img, min_area=15)

    clean_img2 = close_window_edges(clean_img1, dilate_kernel=3,
                                            erode_kernel=3,
                                            dilate_iter=3,
                                            erode_iter=3)
    
    folder = './output/pp_output/'
    os.makedirs(folder, exist_ok=True)
    input_name = os.path.splitext(os.path.basename(room_path))[0].replace('_room','')
    path = f'{folder}{input_name}_{config["segment"]["patch_size"]}_{config["segment"]["stride"]}_clean.png'
    cv2.imwrite(path, cv2.cvtColor(clean_img2, cv2.COLOR_RGB2BGR))

    return path