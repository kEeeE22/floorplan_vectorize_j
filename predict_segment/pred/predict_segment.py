# from .post_process_dpvr import *
# import torch
# import cv2
# import numpy as np
# from torchvision import transforms
# import os

# from ..model.net import DFPmodel
# from ..rgb_ind_convertor import ind2rgb, floorplan_fuse_map, floorplan_boundary_map

# def BCHW2colormap(tensor,earlyexit=False):
#     if tensor.size(0) != 1:
#         tensor = tensor[0].unsqueeze(0)
#     result = tensor.squeeze().permute(1,2,0).cpu().detach().numpy()
#     if earlyexit:
#         return result
#     result = np.argmax(result,axis=2)
#     return result

# def initialize(image_path, loadmodel):
#     # device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(device)
#     # data
#     trans = transforms.Compose([transforms.ToTensor()])
#     orig = cv2.imread(image_path)
#     orig = cv2.resize(orig,(512,512))
#     image = trans(orig.astype(np.float32)/255.)
#     image = image.unsqueeze(0).to(device)
#     # model
#     model = DFPmodel()
#     model.load_state_dict(torch.load(loadmodel, weights_only=True, map_location="cpu"))
#     model.to(device)
#     return device,orig,image,model

# def post_process(rm_ind,bd_ind):
#     hard_c = (bd_ind>0).astype(np.uint8)
#     # region from room prediction 
#     rm_mask = np.zeros(rm_ind.shape)
#     rm_mask[rm_ind>0] = 1
#     # region from close wall line
#     cw_mask = hard_c
#     # regine close wall mask by filling the gap between bright line
#     cw_mask = fill_break_line(cw_mask)

#     fuse_mask = cw_mask + rm_mask
#     fuse_mask[fuse_mask>=1] = 255

#     # refine fuse mask by filling the hole
#     fuse_mask = flood_fill(fuse_mask)
#     fuse_mask = fuse_mask//255

#     # one room one label
#     new_rm_ind = refine_room_region(cw_mask,rm_ind)

#     # ignore the background mislabeling
#     new_rm_ind = fuse_mask*new_rm_ind

#     return new_rm_ind

# def predict_segment(image_path, loadmodel, postprocess=True):
#     device, orig,image,model = initialize(image_path, loadmodel)
#     # run
#     with torch.no_grad():
#         model.eval()
#         logits_r,logits_cw = model(image)
#         predroom = BCHW2colormap(logits_r)
#         predboundary = BCHW2colormap(logits_cw)
#     if postprocess:
#         # postprocess
#         predroom = post_process(predroom,predboundary)
#     rgb = ind2rgb(predroom,color_map=floorplan_fuse_map)
#     predboundary = ind2rgb(predboundary,
#                                 color_map=floorplan_boundary_map)
#     # cv2.imwrite(f'./log/pred/pred_room.png',rgb[:,:,::-1])
#     predboundary = predboundary.astype(np.uint8)

#     #save
#     directory_path = "./output/segment_output/"
#     os.makedirs(directory_path, exist_ok=True)
#     input_name = os.path.splitext(os.path.basename(image_path))[0]
#     suffix = "_post" if postprocess else ""
#     room_path = f'{directory_path}{input_name}{suffix}_room.png'
#     boundary_path = f'{directory_path}{input_name}{suffix}_boundary.png'

#     cv2.imwrite(room_path, rgb[:,:,::-1])
#     cv2.imwrite(boundary_path,cv2.cvtColor(predboundary, cv2.COLOR_RGB2BGR))

#     return room_path, boundary_path


from .post_process_dpvr import *
import torch
import cv2
import numpy as np
from torchvision import transforms
import os
import math
import yaml

from ..model.net import DFPmodel
from ..rgb_ind_convertor import ind2rgb, floorplan_fuse_map, floorplan_boundary_map

with open('./config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def sliding_window_inference(model, image_path, patch_size=512, stride=256, device='cuda', save_patches=False):
    
    image = cv2.imread(image_path)
    h_orig, w_orig = image.shape[:2]
    input_name = os.path.splitext(os.path.basename(image_path))[0]

    trans = transforms.Compose([transforms.ToTensor()])
    
    pad_h = (patch_size - h_orig % stride) % stride + (patch_size - stride)
    pad_w = (patch_size - w_orig % stride) % stride + (patch_size - stride)
    image_padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    h_pad, w_pad = image_padded.shape[:2]

    if save_patches:
        debug_dir = f"./output/patches/{input_name}/"
        os.makedirs(debug_dir, exist_ok=True)
        print(f"DEBUG: Saving patches to {debug_dir}")

    full_logits_r = None 
    full_logits_cw = None
    count_map = np.zeros((h_pad, w_pad), dtype=np.float32)

    model.eval()
    
    with torch.no_grad():
        for y in range(0, h_pad - patch_size + 1, stride):
            for x in range(0, w_pad - patch_size + 1, stride):
                patch = image_padded[y:y+patch_size, x:x+patch_size]

                if save_patches:
                    patch_filename = f"{debug_dir}patch_y{y}_x{x}.png"
                    cv2.imwrite(patch_filename, patch)

                patch_tensor = trans(patch.astype(np.float32)/255.).unsqueeze(0).to(device)
                logits_r, logits_cw = model(patch_tensor)
                
                pred_r = logits_r[0].permute(1, 2, 0).cpu().numpy()
                pred_cw = logits_cw[0].permute(1, 2, 0).cpu().numpy()
                
                if full_logits_r is None:
                    full_logits_r = np.zeros((h_pad, w_pad, pred_r.shape[2]), dtype=np.float32)
                    full_logits_cw = np.zeros((h_pad, w_pad, pred_cw.shape[2]), dtype=np.float32)
                
                full_logits_r[y:y+patch_size, x:x+patch_size] += pred_r
                full_logits_cw[y:y+patch_size, x:x+patch_size] += pred_cw
                count_map[y:y+patch_size, x:x+patch_size] += 1.0

    count_map = np.expand_dims(count_map, axis=-1)
    full_logits_r /= count_map
    full_logits_cw /= count_map
    full_logits_r = full_logits_r[:h_orig, :w_orig]
    full_logits_cw = full_logits_cw[:h_orig, :w_orig]
    
    return full_logits_r, full_logits_cw


def initialize_model_only(loadmodel, device):
    model = DFPmodel()
    model.load_state_dict(torch.load(loadmodel, weights_only=True, map_location="cpu"))
    model.to(device)
    return model

def post_process(rm_ind, bd_ind):
    hard_c = (bd_ind > 0).astype(np.uint8)
    rm_mask = np.zeros(rm_ind.shape)
    rm_mask[rm_ind > 0] = 1
    cw_mask = hard_c
    cw_mask = fill_break_line(cw_mask)
    fuse_mask = cw_mask + rm_mask
    fuse_mask[fuse_mask >= 1] = 255
    fuse_mask = flood_fill(fuse_mask)
    fuse_mask = fuse_mask // 255
    new_rm_ind = refine_room_region(cw_mask, rm_ind)
    new_rm_ind = fuse_mask * new_rm_ind
    return new_rm_ind

def predict_segment(image_path, loadmodel, postprocess=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = initialize_model_only(loadmodel, device)

    patch_size = config['segment']['patch_size']
    stride = config['segment']['stride']
    save_patches = config['segment']['save_patches']
    print("Running sliding window inference...")
    logits_r, logits_cw = sliding_window_inference(model, image_path, patch_size=patch_size, stride=stride, device=device, save_patches=save_patches)

    predroom = np.argmax(logits_r, axis=2)      # (H, W)
    predboundary = np.argmax(logits_cw, axis=2) # (H, W)     
    if postprocess:
        print("Post-processing...")
        predroom = post_process(predroom, predboundary)

    rgb = ind2rgb(predroom, color_map=floorplan_fuse_map)
    predboundary_rgb = ind2rgb(predboundary, color_map=floorplan_boundary_map)
    predboundary_rgb = predboundary_rgb.astype(np.uint8)

    directory_path = "./output/segment_output/"
    os.makedirs(directory_path, exist_ok=True)
    input_name = os.path.splitext(os.path.basename(image_path))[0]
    suffix = "_post" if postprocess else ""
    room_path = f'{directory_path}{input_name}{suffix}_{patch_size}_{stride}_room.png'
    boundary_path = f'{directory_path}{input_name}{suffix}_{patch_size}_{stride}_boundary.png'

    cv2.imwrite(room_path, rgb[:,:,::-1])
    cv2.imwrite(boundary_path, cv2.cvtColor(predboundary_rgb, cv2.COLOR_RGB2BGR))
    
    print(f"Saved to: {room_path}")
    return room_path, boundary_path