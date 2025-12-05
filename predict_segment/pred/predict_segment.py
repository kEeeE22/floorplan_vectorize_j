from .post_process_dpvr import *
import torch
import cv2
import numpy as np
from torchvision import transforms
import os

from ..model.net import DFPmodel
from ..rgb_ind_convertor import ind2rgb, floorplan_fuse_map, floorplan_boundary_map

def BCHW2colormap(tensor,earlyexit=False):
    if tensor.size(0) != 1:
        tensor = tensor[0].unsqueeze(0)
    result = tensor.squeeze().permute(1,2,0).cpu().detach().numpy()
    if earlyexit:
        return result
    result = np.argmax(result,axis=2)
    return result

def initialize(image_path, loadmodel):
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # data
    trans = transforms.Compose([transforms.ToTensor()])
    orig = cv2.imread(image_path)
    orig = cv2.resize(orig,(512,512))
    image = trans(orig.astype(np.float32)/255.)
    image = image.unsqueeze(0).to(device)
    # model
    model = DFPmodel()
    model.load_state_dict(torch.load(loadmodel, weights_only=True, map_location="cpu"))
    model.to(device)
    return device,orig,image,model

def post_process(rm_ind,bd_ind):
    hard_c = (bd_ind>0).astype(np.uint8)
    # region from room prediction 
    rm_mask = np.zeros(rm_ind.shape)
    rm_mask[rm_ind>0] = 1
    # region from close wall line
    cw_mask = hard_c
    # regine close wall mask by filling the gap between bright line
    cw_mask = fill_break_line(cw_mask)

    fuse_mask = cw_mask + rm_mask
    fuse_mask[fuse_mask>=1] = 255

    # refine fuse mask by filling the hole
    fuse_mask = flood_fill(fuse_mask)
    fuse_mask = fuse_mask//255

    # one room one label
    new_rm_ind = refine_room_region(cw_mask,rm_ind)

    # ignore the background mislabeling
    new_rm_ind = fuse_mask*new_rm_ind

    return new_rm_ind

def predict_segment(image_path, loadmodel, postprocess=True):
    device, orig,image,model = initialize(image_path, loadmodel)
    # run
    with torch.no_grad():
        model.eval()
        logits_r,logits_cw = model(image)
        predroom = BCHW2colormap(logits_r)
        predboundary = BCHW2colormap(logits_cw)
    if postprocess:
        # postprocess
        predroom = post_process(predroom,predboundary)
    rgb = ind2rgb(predroom,color_map=floorplan_fuse_map)
    predboundary = ind2rgb(predboundary,
                                color_map=floorplan_boundary_map)
    # cv2.imwrite(f'./log/pred/pred_room.png',rgb[:,:,::-1])
    predboundary = predboundary.astype(np.uint8)

    #save
    directory_path = "./output/segment_output/"
    os.makedirs(directory_path, exist_ok=True)
    input_name = os.path.splitext(os.path.basename(image_path))[0]
    suffix = "_post" if postprocess else ""
    room_path = f'{directory_path}{input_name}{suffix}_room.png'
    boundary_path = f'{directory_path}{input_name}{suffix}_boundary.png'

    cv2.imwrite(room_path, rgb[:,:,::-1])
    cv2.imwrite(boundary_path,cv2.cvtColor(predboundary, cv2.COLOR_RGB2BGR))

    return room_path, boundary_path