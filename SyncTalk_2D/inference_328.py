import argparse
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from unet_328 import Model
from tqdm import tqdm
from utils import (AudioEncoder, AudDataset, get_audio_features,
                   apply_mouth_mask, read_mask_version, suppress_invented_chroma)
# from unet2 import Model
# from unet_att import Model

import time
import shutil  # Added for Windows file operations

parser = argparse.ArgumentParser(description='Train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--asr', type=str, default="ave")
parser.add_argument('--name', type=str, default="May")
parser.add_argument('--audio_path', type=str, default="demo/talk_hb.wav")
parser.add_argument('--start_frame', type=int, default=0)
parser.add_argument('--parsing', type=bool, default=False)
parser.add_argument('--ssl_model', type=str, default="facebook/wav2vec2-xls-r-300m",
                    help="Encoder for --asr ssl. Must match what training used.")
parser.add_argument('--ssl_layer', type=int, default=12,
                    help="Hidden layer for --asr ssl. Must match what training used.")
parser.add_argument('--ref_frame', type=int, default=-1,
                    help="Frame index for the appearance reference (channels 0-2). "
                         "-1 = auto-pick the most closed-mouth frame.")
parser.add_argument('--no_chroma_fix', action='store_true',
                    help="Skip the post-hoc chroma correction. The sync loss leaves a blue "
                         "smear in dark regions; this removes it without retraining.")
parser.add_argument('--no_fixed_ref', action='store_true',
                    help="Revert to the original behaviour where channels 0-2 follow the "
                         "current frame. Leaks the source mouth; use only for A/B.")
args = parser.parse_args()

checkpoint_path = os.path.join(".", "checkpoint", args.name)
# Get the latest numbered checkpoint. Only epoch-numbered files are plain
# state_dicts; last.pth is a resume bundle (model+optimizer+scaler) and would
# both break int() here and fail load_state_dict below.
checkpoint_files = [f for f in os.listdir(checkpoint_path)
                    if f.endswith('.pth') and os.path.splitext(f)[0].isdigit()]
if not checkpoint_files:
    raise FileNotFoundError(
        f"No epoch-numbered .pth in {checkpoint_path}. "
        "Training saves those every 5 epochs; last.pth alone is not usable here.")
checkpoint = os.path.join(checkpoint_path, sorted(checkpoint_files, key=lambda x: int(x.split(".")[0]))[-1])
print(checkpoint)

# The mask must match what this checkpoint trained with, or the model sees an
# input it has never encountered. Legacy checkpoints predate train_config.json.
MASK_VERSION = read_mask_version(checkpoint)
print(f"Mouth mask: {MASK_VERSION}" +
      ("  (legacy - jaw visible, so mouth shape partly leaks from the source video)"
       if MASK_VERSION == "legacy" else "  (jaw hidden)"))

# Extract audio filename without extension
audio_filename = os.path.basename(args.audio_path)
audio_name_without_ext = os.path.splitext(audio_filename)[0]
checkpoint_name = os.path.splitext(os.path.basename(checkpoint))[0]

save_path = os.path.join(".", "result", f"{args.name}_{audio_name_without_ext}_{checkpoint_name}.mp4")
temp_save_path = save_path.replace(".mp4", "_temp.mp4")
# result/ is gitignored, so it does not exist after a fresh clone. cv2.VideoWriter
# fails SILENTLY when the directory is missing - write() becomes a no-op and the
# script still exits 0 with no video.
os.makedirs(os.path.dirname(save_path), exist_ok=True)
dataset_dir = os.path.join(".", "dataset", args.name)
audio_path = args.audio_path
mode = args.asr


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if mode == "ssl":
    # Bangla SSL encoder. Same output convention as the ave path below
    # (25 fps rows, first/last duplicated), so everything downstream matches.
    from extract_ssl_features import extract as extract_ssl
    # Reuse the normalisation the model was trained with. Recomputing per-clip
    # stats on new audio would feed the model differently-scaled features.
    stats = None
    stats_path = os.path.join(dataset_dir, "aud_ssl_stats.npz")
    if os.path.exists(stats_path):
        z = np.load(stats_path, allow_pickle=True)
        stats = (z["mean"], z["std"])
        print(f"using training-time SSL stats from {stats_path} (layer {z['layer']})")
    else:
        print(f"WARNING: {stats_path} not found - normalising on this clip instead. "
              "Scores will be slightly off unless the clip is long.")
    audio_feats = extract_ssl(audio_path, model_name=args.ssl_model,
                              layer=args.ssl_layer, device=device, fp16=True, stats=stats)
else:
    model = AudioEncoder().to(device).eval()
    ckpt = torch.load(os.path.join(".", "model", "checkpoints", "audio_visual_encoder.pth"))
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
img_dir = os.path.join(dataset_dir, "full_body_img")
lms_dir = os.path.join(dataset_dir, "landmarks")
len_img = len([f for f in os.listdir(img_dir) if f.endswith('.jpg')]) - 1
exm_img = cv2.imread(os.path.join(img_dir, "0.jpg"))
h, w = exm_img.shape[:2]
if args.parsing:
    parsing_dir = os.path.join(dataset_dir, "parsing")

if mode=="hubert" or mode=="ave" or mode=="ssl":
    video_writer = cv2.VideoWriter(temp_save_path, cv2.VideoWriter_fourcc(*'MJPG'), 25, (w, h))
if mode=="wenet":
    video_writer = cv2.VideoWriter(temp_save_path, cv2.VideoWriter_fourcc(*'MJPG'), 20, (w, h))
if not video_writer.isOpened():
    raise RuntimeError(
        f"cv2.VideoWriter could not open {temp_save_path}. Check the directory exists "
        "and that OpenCV has an MJPG encoder.")
step_stride = 0
img_idx = 0
chroma_fixed_px = []

net = Model(6, mode).cuda()
net.load_state_dict(torch.load(checkpoint))
net.eval()

# --- Fixed appearance reference (channels 0-2) ------------------------------
# Training (datasetsss_328.py) puts a RANDOM DIFFERENT frame in channels 0-2 and
# the current frame, mouth blacked out, in channels 3-5. Inference used to put
# the CURRENT frame in channels 0-2 with its mouth visible - so the reference
# carried the source video's real mouth, and since the reference walks through
# footage of the person talking, the generated mouth followed it. That is why a
# silent audio track still produced a talking avatar.
#
# Holding the reference on one closed-mouth frame restores the training
# distribution and leaves audio as the only varying driver of the mouth.
INNER_LIP_UPPER, INNER_LIP_LOWER = [103, 104, 105], [107, 108, 109]

def _read_lms(p):
    return np.array([np.array(l.split(" "), dtype=np.float32)
                     for l in open(p).read().splitlines()])

def _crop_320(frame_idx):
    im = cv2.imread(os.path.join(img_dir, f"{frame_idx}.jpg"))
    lm = _read_lms(os.path.join(lms_dir, f"{frame_idx}.lms")).astype(np.int32)
    x0, y0 = max(0, int(lm[1][0])), max(0, int(lm[52][1]))
    x1 = min(im.shape[1] - 1, int(lm[31][0]))
    y1 = min(im.shape[0] - 1, y0 + (x1 - x0))
    c = cv2.resize(im[y0:y1, x0:x1], (328, 328), interpolation=cv2.INTER_CUBIC)
    return c[4:324, 4:324].copy()

fixed_ref_T = None
if not args.no_fixed_ref:
    ref_idx = args.ref_frame
    if ref_idx < 0:                       # auto: the most closed mouth available
        aps = []
        for i in range(len_img):
            p = os.path.join(lms_dir, f"{i}.lms")
            if not os.path.exists(p):
                continue
            L = _read_lms(p)
            aps.append((float(L[INNER_LIP_LOWER, 1].mean() - L[INNER_LIP_UPPER, 1].mean()), i))
        if not aps:
            raise RuntimeError("No landmarks found; pass --ref_frame explicitly.")
        ap_min, ref_idx = min(aps)
        print(f"[Ref] Auto-selected frame {ref_idx} as appearance reference "
              f"(mouth aperture {ap_min:.2f}; median {np.median([a for a,_ in aps]):.1f})")
    else:
        print(f"[Ref] Using frame {ref_idx} as appearance reference")
    _r = _crop_320(ref_idx).transpose(2, 0, 1).astype(np.float32) / 255.0
    fixed_ref_T = torch.from_numpy(_r)
else:
    print("[Ref] --no_fixed_ref: channels 0-2 follow the current frame "
          "(original behaviour; mouth can move even on silent audio)")
for i in tqdm(range(audio_feats.shape[0])):
    if img_idx>len_img - 1:
        step_stride = -1
    if img_idx<1:
        step_stride = 1
    img_idx += step_stride
    img_path = os.path.join(img_dir, f"{img_idx+args.start_frame}.jpg")
    if args.parsing:  # Read semantic segmentation map, use ori img for [0, 0, 255] area, not pred result
        parsing_path = os.path.join(parsing_dir, f"{img_idx+args.start_frame}.png")
        parsing = cv2.imread(parsing_path)
        # print(parsing.shape)

    # print(img_path)
    lms_path = os.path.join(lms_dir, f"{img_idx+args.start_frame}.lms")
    
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
    # ymax = lms[16][1] + width//15
    # ymax = ymin + width//7*6
    crop_img = img[ymin:ymax, xmin:xmax]  
    crop_img_par = crop_img.copy()  
    if args.parsing:  # Read semantic segmentation map, use ori img for [0, 0, 255] area, not pred result
        crop_parsing_img = parsing[ymin:ymax, xmin:xmax] 
        # crop_parsing_img = cv2.resize(crop_parsing_img, (328, 328), cv2.INTER_AREA)
    h, w = crop_img.shape[:2]
    crop_img = cv2.resize(crop_img, (328, 328), interpolation=cv2.INTER_CUBIC)
    crop_img_ori = crop_img.copy()
    img_real_ex = crop_img[4:324, 4:324].copy()
    img_real_ex_ori = img_real_ex.copy()
    # if args.parsing:
        # img_real_ex_ori_ori = img_real_ex.copy()
    img_masked = apply_mouth_mask(img_real_ex_ori, MASK_VERSION)
    img_masked = img_masked.transpose(2,0,1).astype(np.float32)
    img_real_ex = img_real_ex.transpose(2,0,1).astype(np.float32)
    
    # Channels 0-2: fixed appearance reference (see note above), not the
    # current frame - otherwise the reference's own mouth drives the output.
    img_real_ex_T = (fixed_ref_T if fixed_ref_T is not None
                     else torch.from_numpy(img_real_ex / 255.0))
    img_masked_T = torch.from_numpy(img_masked / 255.0)
    img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=0)[None]
    
    audio_feat = get_audio_features(audio_feats, i)
    if mode=="hubert":
        audio_feat = audio_feat.reshape(32,32,32)
    if mode=="wenet":
        audio_feat = audio_feat.reshape(256,16,32)
    if mode=="ave":
        audio_feat = audio_feat.reshape(32,16,16)
    if mode=="ssl":
        # 16 frames x 1024 dims = 16384 = 16*32*32
        audio_feat = audio_feat.reshape(16,32,32)
    audio_feat = audio_feat[None]
    audio_feat = audio_feat.cuda()
    img_concat_T = img_concat_T.cuda()
    
    with torch.no_grad():
        pred = net(img_concat_T, audio_feat)[0]
        
    pred = pred.cpu().numpy().transpose(1,2,0)*255
    pred = np.array(pred, dtype=np.uint8)
    if not args.no_chroma_fix:
        # The sync loss leaves a blue smear in the beard; see utils for why.
        # Compares against the real crop, so the subject's blue shirt is safe.
        pred, changed = suppress_invented_chroma(pred, crop_img_ori[4:324, 4:324])
        chroma_fixed_px.append(changed)
    # if args.parsing:  # Read semantic segmentation map, use ori img for [0, 0, 255] area, not pred result
        # parsing_mask = (crop_parsing_img[4:324, 4:324] == [0, 0, 255]).all(axis=2)
        # pred[parsing_mask] = img_real_ex_ori_ori[parsing_mask]
    crop_img_ori[4:324, 4:324] = pred
    crop_img_ori = cv2.resize(crop_img_ori, (w, h), interpolation=cv2.INTER_CUBIC)
    if args.parsing:  # Read semantic segmentation map, use ori img for [0, 0, 255] and [255, 255, 255] area, not pred result
        parsing_mask = (crop_parsing_img == [0, 0, 255]).all(axis=2) | (crop_parsing_img == [255, 255, 255]).all(axis=2)
        crop_img_ori[parsing_mask] = crop_img_par[parsing_mask]
    img[ymin:ymax, xmin:xmax] = crop_img_ori
    # y_gap = lms[16][1] - lms[52][1] + h//10
    # print(y_gap, h, h//10, width)
    # crop_img_ori = crop_img_ori[:y_gap,:]
    # cv2.imwrite(f"./temp/{i}.jpg", crop_img_ori)
    # img[ymin:ymin+y_gap, xmin:xmax] = crop_img_ori
    video_writer.write(img)
video_writer.release()

# Use ffmpeg command compatible with Windows
# Use absolute paths with proper escaping
temp_save_path_abs = os.path.abspath(temp_save_path)
audio_path_abs = os.path.abspath(audio_path)
save_path_abs = os.path.abspath(save_path)

# For Windows, we need to properly escape the paths
ffmpeg_cmd = f'ffmpeg -i "{temp_save_path_abs}" -i "{audio_path_abs}" -c:v libx264 -c:a aac -crf 20 "{save_path_abs}" -y'
# os.system does not raise on failure, so an ffmpeg error used to leave the
# script exiting 0 with no output file. Check it.
rc = os.system(ffmpeg_cmd)
if rc != 0 or not os.path.exists(save_path_abs):
    raise RuntimeError(
        f"ffmpeg muxing failed (exit {rc}) - no video at {save_path_abs}.\n"
        f"Command: {ffmpeg_cmd}")

# Remove temporary file
try:
    os.remove(temp_save_path_abs)
except OSError as e:
    print(f"Error removing temporary file: {e}")

if chroma_fixed_px:
    import numpy as _np
    print(f"[INFO] chroma correction touched {_np.mean(chroma_fixed_px)*100:.3f}% of generated pixels on average")
print(f"[INFO] ===== save video to {save_path_abs} "
      f"({os.path.getsize(save_path_abs)/1024**2:.1f} MB) =====")