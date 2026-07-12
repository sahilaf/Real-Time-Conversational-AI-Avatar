import argparse
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from unet import Model
from tqdm import tqdm
from utils import AudioEncoder, AudDataset, get_audio_features
# from unet2 import Model
# from unet_att import Model

import time
parser = argparse.ArgumentParser(description='Train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--asr', type=str, default="ave")
parser.add_argument('--name', type=str, default="")
parser.add_argument('--audio_path', type=str, default="")
parser.add_argument('--start_frame', type=int, default=0)
args = parser.parse_args()

checkpoint_path = os.path.join("./checkpoint", args.name)
checkpoint = os.path.join(checkpoint_path, sorted(os.listdir(checkpoint_path), key=lambda x: int(x.split(".")[0]))[-1])
print(checkpoint)
save_path = os.path.join("./result", args.name+"_"+args.audio_path.split("/")[-1].split(".")[0]+"_"+checkpoint.split("/")[-1].split(".")[0]+".mp4")
dataset_dir = os.path.join("./dataset", args.name)
audio_path = args.audio_path
mode = args.asr


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AudioEncoder().to(device).eval()
ckpt = torch.load('model/checkpoints/audio_visual_encoder.pth')
model.load_state_dict({f'audio_encoder.{k}': v for k, v in ckpt.items()})
dataset = AudDataset(audio_path)
data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
outputs = []
for mel in data_loader:
    mel = mel.to(device)
    with torch.no_grad():
        out = model(mel)
    outputs.append(out)
outputs = torch.cat(outputs, dim=0).cpu()
first_frame, last_frame = outputs[:1], outputs[-1:]
audio_feats = torch.cat([first_frame.repeat(1, 1), outputs, last_frame.repeat(1, 1)],
                            dim=0).numpy()
img_dir = os.path.join(dataset_dir, "full_body_img/")
lms_dir = os.path.join(dataset_dir, "landmarks/")
len_img = len(os.listdir(img_dir)) - 1
exm_img = cv2.imread(img_dir+"0.jpg")
h, w = exm_img.shape[:2]

if mode=="hubert" or mode=="ave":
    video_writer = cv2.VideoWriter(save_path.replace(".mp4", "temp.mp4"), cv2.VideoWriter_fourcc('M','J','P', 'G'), 25, (w, h))
if mode=="wenet":
    video_writer = cv2.VideoWriter(save_path.replace(".mp4", "temp.mp4"), cv2.VideoWriter_fourcc('M','J','P', 'G'), 20, (w, h))
step_stride = 0
img_idx = 0

net = Model(6, mode).cuda()
net.load_state_dict(torch.load(checkpoint))
net.eval()
for i in tqdm(range(audio_feats.shape[0])):
    if img_idx>len_img - 1:
        step_stride = -1
    if img_idx<1:
        step_stride = 1
    img_idx += step_stride
    img_path = img_dir + str(img_idx+args.start_frame)+'.jpg'
    # print(img_path)
    lms_path = lms_dir + str(img_idx+args.start_frame)+'.lms'
    
    img = cv2.imread(img_path)
    lms_list = []
    with open(lms_path, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            arr = line.split(" ")
            arr = np.array(arr, dtype=np.float32)
            lms_list.append(arr)
    lms = np.array(lms_list, dtype=np.int32)
    xmin = lms[1][0]
    ymin = lms[52][1]

    xmax = lms[31][0]
    width = xmax - xmin
    ymax = ymin + width
    crop_img = img[ymin:ymax, xmin:xmax]
    h, w = crop_img.shape[:2]
    crop_img = cv2.resize(crop_img, (168, 168), cv2.INTER_AREA)
    crop_img_ori = crop_img.copy()
    img_real_ex = crop_img[4:164, 4:164].copy()
    img_real_ex_ori = img_real_ex.copy()
    img_masked = cv2.rectangle(img_real_ex_ori,(5,5,150,145),(0,0,0),-1)
    
    img_masked = img_masked.transpose(2,0,1).astype(np.float32)
    img_real_ex = img_real_ex.transpose(2,0,1).astype(np.float32)
    
    img_real_ex_T = torch.from_numpy(img_real_ex / 255.0)
    img_masked_T = torch.from_numpy(img_masked / 255.0)
    img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=0)[None]
    
    audio_feat = get_audio_features(audio_feats, i)
    if mode=="hubert":
        audio_feat = audio_feat.reshape(32,32,32)
    if mode=="wenet":
        audio_feat = audio_feat.reshape(256,16,32)
    if mode=="ave":
        audio_feat = audio_feat.reshape(32,16,16)
    audio_feat = audio_feat[None]
    audio_feat = audio_feat.cuda()
    img_concat_T = img_concat_T.cuda()
    
    with torch.no_grad():
        pred = net(img_concat_T, audio_feat)[0]
        
    pred = pred.cpu().numpy().transpose(1,2,0)*255
    pred = np.array(pred, dtype=np.uint8)
    crop_img_ori[4:164, 4:164] = pred
    crop_img_ori = cv2.resize(crop_img_ori, (w, h))
    # img[ymin:ymin + int(width/3*2), xmin:xmax] = crop_img_ori[:int(width/3*2), :]
    img[ymin:ymax, xmin:xmax] = crop_img_ori
    video_writer.write(img)
video_writer.release()

# os.system(f"ffmpeg -i {save_path.replace('.mp4', 'temp.mp4')} -i {audio_path} -c:v libx264 -c:a aac {save_path}")
os.system(f"ffmpeg -i {save_path.replace('.mp4', 'temp.mp4')} -i {audio_path} -c:v libx264 -c:a aac -crf 10 {save_path}")
os.system(f"rm {save_path.replace('.mp4', 'temp.mp4')}")
print(f"[INFO] ===== save video to {save_path} =====")