import torch
import torch.nn as nn
import numpy as np
import openslide
import h5py
import os

from torch.utils.data import DataLoader
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from patchdataset import Roi_Seg_Dataset, Patch_Seg_Dataset
from models.extractor import resnet50
import models.ResNet as ResNet
from models.ccl import CCL
from models.ctran import ctranspath
from models.simclr_ciga import simclr_ciga_model

from utils.file_utils import save_hdf5
from utils.utils import collate_features
from PIL import Image
from tqdm import tqdm
import h5py
import openslide
import argparse


def extract_feats(args, h5_file_path, wsi, slide_path, model, output_path, target_roi_size=2048, patch_size=256, levels=[0,1,2], batch_size=1, is_stain_norm=False):
    """
    Extracts features from the given Whole Slide Image (WSI) using the specified model.
    
    Args:
        args: Parsed command line arguments.
        h5_file_path (str): Path to the HDF5 file containing patch data.
        wsi: OpenSlide object representing the WSI.
        slide_path (str): Path to the WSI file.
        model: Pre-trained model for feature extraction.
        output_path (str): Path to save the extracted features.
        target_roi_size (int, optional): Target size for ROI. Default is 2048.
        patch_size (int, optional): Size of each patch. Default is 256.
        levels (list, optional): List of levels to process. Default is [0,1,2].
        batch_size (int, optional): Batch size for processing. Default is 1.
        is_stain_norm (bool, optional): Whether to perform stain normalization. Default is False.
    """
    if args.pretrained_model == 'ctranspath':
        roi_dataset = Roi_Seg_Dataset(args.pretrained_model, h5_file_path, slide_path, wsi, levels, target_roi_size, patch_size, is_stain_norm, resize=True)
    else:
        roi_dataset = Roi_Seg_Dataset(args.pretrained_model, h5_file_path, slide_path, wsi, levels, target_roi_size, patch_size, is_stain_norm)
    roi_dataloader = DataLoader(roi_dataset, batch_size=batch_size, num_workers=4)
    mode = 'w'

    for batch, coords, available in tqdm(roi_dataloader):
        with torch.no_grad():
            for b in range(batch_size):
                if not available[b]:
                    continue
                img_batch = batch[b].cuda()
                features = model(img_batch)
                features = features.unsqueeze(0)
                features = features.cpu().numpy()
                coord = coords[b].unsqueeze(0).numpy()
                asset_dict = {'features': features, 'coords': coord}
                save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
                mode = 'a'


def extract_feats_patch(h5_file_path, wsi, slide_path, model, output_path, patch_size=256, batch_size=1, is_stain_norm=False):
    """
    Extracts patch-based features from a Whole Slide Image (WSI).
    
    Args:
        h5_file_path (str): Path to the HDF5 file containing patch data.
        wsi: OpenSlide object representing the WSI.
        slide_path (str): Path to the WSI file.
        model: Pre-trained model for feature extraction.
        output_path (str): Path to save the extracted features.
        patch_size (int, optional): Size of each patch. Default is 256.
        batch_size (int, optional): Batch size for processing. Default is 1.
        is_stain_norm (bool, optional): Whether to perform stain normalization. Default is False.
    """
    roi_dataset = Patch_Seg_Dataset(h5_file_path, slide_path, wsi, patch_size, is_stain_norm)
    roi_dataloader = DataLoader(roi_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, collate_fn=collate_features)
    mode = 'w'
    
    for batch, coords, available in tqdm(roi_dataloader):
        with torch.no_grad():
            batch = batch.cuda()
            features = model(batch)
            features = features.cpu().numpy()
            if features.shape[0] < 2:
                continue
            features_normal = features[available]
            coords_normal = coords[available]
            if features_normal.shape[0] > 0:
                asset_dict = {'features': features_normal, 'coords': coords_normal}
                save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
                mode = 'a'


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default='/data/COMPL/WSI/CLAM')
parser.add_argument('--data_slide_dir', type=str, default='/data/COMPL/WSI/HE')
parser.add_argument('--csv_path', type=str, default='/data/COMPL/WSI/data_csv/PDAC.csv')
parser.add_argument('--dataset', type=str, default='PDAC')
parser.add_argument('--data_format', type=str, default='roi', choices=['roi','patch'])
parser.add_argument('--feat_dir', type=str, default='/data/COMPL/WSI/FT')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=256)
parser.add_argument('--target_roi_size', type=int, default=2048)
parser.add_argument('--level', default=1, type=int, choices=[0,1,2])
parser.add_argument('--is_stain_norm', action='store_true', default=False, help='whether stain normalization')
parser.add_argument('--pretrained_model', type=str, default='ImageNet', choices=['ImageNet', 'RetCCL', 'simclr-ciga', 'ctranspath'], help='model weights for extracting features')
args = parser.parse_args()


if __name__ == '__main__':
    os.makedirs(args.feat_dir, exist_ok=True)
    if args.is_stain_norm:
        args.feat_dir = os.path.join(args.feat_dir, f'feats_{args.pretrained_model}_norm')
    else:
        args.feat_dir = os.path.join(args.feat_dir, f'feats_{args.pretrained_model}')
    os.makedirs(args.feat_dir, exist_ok=True)
    dest_files = os.listdir(args.feat_dir)
    data_csv = pd.read_csv(args.csv_path)
    slide_id = data_csv['slide_id'].values
    slide_path = data_csv['path'].values
    roi_size_list = [2048, 1024, 512]
    levels = [i for i in range(4 - args.level)]
    target_roi_size = roi_size_list[args.level]
    
    if args.pretrained_model == 'ImageNet':
        model = resnet50(pretrained=True).cuda()
    elif args.pretrained_model == 'RetCCL':
        backbone = ResNet.resnet50
        model = CCL(backbone, 128, 65536, mlp=True, two_branch=True, normlinear=True).cuda()
        ckpt_path = f'models/{args.pretrained_model}_ckpt.pth'
        model.load_state_dict(torch.load(ckpt_path), strict=True)
        model.encoder_q.fc = nn.Identity()
        model.encoder_q.instDis = nn.Identity()
        model.encoder_q.groupDis = nn.Identity()
    elif args.pretrained_model == 'ctranspath':
        model = ctranspath().cuda()
        model.head = nn.Identity()
        td = torch.load(r'models/ctranspath.pth')
        model.load_state_dict(td['model'], strict=True)
    else:
        model = simclr_ciga_model().cuda()

    model.eval()

    for i in range(len(slide_id)):
        print(f'extract features from {slide_id[i]},{i}/{len(slide_id)}')

        bag_name = slide_id[i]+'.h5'
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)

        slide_file_path = args.data_slide_dir + slide_path[i]

        if not args.no_auto_skip and slide_id[i]+'.h5' in dest_files:
            print(f'skipped {slide_id[i]}')
            continue
        
        output_path = os.path.join(args.feat_dir, bag_name)
        wsi = openslide.open_slide(slide_file_path)

        if args.data_format == 'roi':
            extract_feats(args,h5_file_path,wsi,slide_file_path,model,output_path,target_roi_size=target_roi_size,levels = levels,is_stain_norm=args.is_stain_norm)
        else:
            extract_feats_patch(h5_file_path,wsi,slide_file_path,model,output_path,batch_size = args.batch_size,is_stain_norm=args.is_stain_norm)


    