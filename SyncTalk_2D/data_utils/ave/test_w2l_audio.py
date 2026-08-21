import sys
import torch
from glob import glob
import os, random, cv2
import numpy as np
import torch.nn as nn
from os.path import dirname, join, basename, isfile
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle as pkl
import collections
import argparse

# models.py is not present in this repo - AudioEncoder lives in the root
# utils.py, and is the same class this script expects (it loads
# data_utils/ave/checkpoints/audio_encoder.pth, byte-identical to
# model/checkpoints/audio_visual_encoder.pth, with the same
# "audio_encoder." key prefixing). Python puts this file's own directory on
# sys.path, so the repo root has to be added explicitly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils import AudioEncoder
from hparams import hparams
import audio

syncnet_T = 5
syncnet_mel_step_size = 16

parser = argparse.ArgumentParser()
parser.add_argument('--wav_path', type=str,
                    default='obama', help='input audio path')

args = parser.parse_args()

class AudDataset(object):
    def __init__(self, wavpath):
        wav = audio.load_wav(wavpath, hparams.sample_rate)

        self.orig_mel = audio.melspectrogram(wav).T
        self.data_len = int( (self.orig_mel.shape[0] - syncnet_mel_step_size) / 80. * float(hparams.fps)) + 2

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_id):

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            window_fnames.append(frame_id)
        return window_fnames

    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (hparams.img_size, hparams.img_size))
            except Exception as e:
                return None

            window.append(img)

        return window

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))
        
        end_idx = start_idx + 16
        if end_idx > spec.shape[0]:
            # print(end_idx, spec.shape[0])
            end_idx = spec.shape[0]
            start_idx = end_idx - 16

        return spec[start_idx: end_idx, :]

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        assert syncnet_T == 5
        start_frame_num = self.get_frame_id(start_frame) + 1 # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels

    def prepare_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        window_idxes = self.get_window(idx)

        # window = self.all_exps[window_idxes]

        # No .copy() here. crop_audio_window only ever slices, never mutates,
        # so copying the whole spectrogram per frame bought nothing and cost
        # O(n) memcpy on every one of n frames - quadratic in video length. A
        # 2-minute clip absorbed it; a 35-minute one copies ~2.8 TB.
        mel = self.crop_audio_window(self.orig_mel, idx)
        # print("mel.shape: ", mel.shape)

        if (mel.shape[0] != syncnet_mel_step_size):
            raise Exception('mel.shape[0] != syncnet_mel_step_size')
        # x = window.float()
        # x = torch.FloatTensor(x)
        mel = torch.FloatTensor(mel.T).unsqueeze(0)
        # indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)

        return mel


device = torch.device('cuda')
model = AudioEncoder()
# ckpt = torch.load('checkpoints/audio_encoder.pth')
ckpt = torch.load('data_utils/ave/checkpoints/audio_encoder.pth')
new_state_dict = collections.OrderedDict()
for key, value in ckpt.items():
   new_state_dict['audio_encoder.' + key] = value
model.load_state_dict(new_state_dict)
model = model.to(device).eval()

dataset = AudDataset(args.wav_path)
save_path = args.wav_path.replace('.wav', '_ave.npy')
data_loader = DataLoader(dataset, batch_size=64, shuffle=False)

print(f"[AVE] {dataset.data_len} frames from {os.path.basename(args.wav_path)}")
outputs = []
with torch.no_grad():
    # tqdm because a long file is otherwise a silent black box, and this stage
    # previously failed without leaving any trace of how far it got
    for mel in tqdm(data_loader, desc="audio features"):
        mel = mel.to(device)
        # .cpu() per batch: keeping every activation on a 4 GB card alongside
        # whatever else is running is needless pressure for no speed gain
        outputs.append(model(mel).cpu())
outputs = torch.cat(outputs, dim=0)
first_frame = outputs[0]
last_frame = outputs[-1]
outputs = torch.cat((first_frame.unsqueeze(0).repeat(1, 1), outputs, last_frame.unsqueeze(0).repeat(1, 1)), dim=0)
print("outputs.shape: ", outputs.shape)
# torch.save(outputs, save_path)
np.save(save_path, outputs.numpy())
