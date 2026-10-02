import os
import cv2
import torch
import random
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils import apply_mouth_mask, MASK_V2

class MyDataset(Dataset):

    def __init__(self, img_dir, mode, mask_version=MASK_V2, start=0, end=None,
                 ranges=None, cache=False, cache_threads=8):
        """start/end bound the frames this dataset may use, inclusive.

        Pass the manifest's train split. Without a bound the model trains on
        EVERY frame in the directory, including the held-out test split it is
        later scored on, and its reconstruction numbers are meaningless next to
        person-generic baselines that have never seen the video at all.

        ranges: list of [start, end] (inclusive) instead of start/end, for a
        train split that surrounds a held-out stretch.

        cache: crop every frame once up front and keep the 320x320 crops in
        RAM (~0.3 MB each). The crop depends only on the frame and its
        landmarks, so samples are pixel-identical to the uncached path; what
        goes away is decoding two 1080p JPEGs per sample every epoch, which
        is what kept the GPU waiting.
        """

        self.img_path_list = []
        self.lms_path_list = []
        self.mode = mode
        # Must match what inference and evaluation use - see utils.apply_mouth_mask
        self.mask_version = mask_version

        for i in range(len(os.listdir(img_dir+"/full_body_img/"))):

            img_path = os.path.join(img_dir+"/full_body_img/", str(i)+".jpg")
            lms_path = os.path.join(img_dir+"/landmarks/", str(i)+".lms")
            self.img_path_list.append(img_path)
            self.lms_path_list.append(lms_path)
        
        if self.mode == "wenet":
            self.audio_feats = np.load(img_dir+"/aud_wenet.npy")
        if self.mode == "hubert":
            self.audio_feats = np.load(img_dir+"/aud_hu.npy")
        if self.mode == "ave":
            self.audio_feats = np.load(img_dir+"/aud_ave.npy")
        if self.mode == "ssl":
            self.audio_feats = np.load(img_dir+"/aud_ssl.npy")


        self.audio_feats = self.audio_feats.astype(np.float32)

        # Usable indices are bounded by BOTH the frames and the audio features -
        # the audio track runs a frame or two short of the video.
        last = min(len(self.img_path_list), self.audio_feats.shape[0] - 1) - 1
        if ranges:
            idx = set()
            for s0, e0 in ranges:
                idx.update(range(max(0, int(s0)), min(int(e0), last) + 1))
            self.indices = sorted(idx)
            if not self.indices:
                raise ValueError(f"empty frame ranges {ranges} (usable 0..{last})")
            lo, hi = self.indices[0], self.indices[-1]
        else:
            lo = max(0, int(start))
            hi = last if end is None else min(int(end), last)
            if hi < lo:
                raise ValueError(f"empty frame range {start}..{end} (usable 0..{last})")
            self.indices = list(range(lo, hi + 1))

        print(img_dir)
        print(self.audio_feats.shape)
        print(len(self.img_path_list))
        if ranges:
            print(f"Frames in {len(ranges)} ranges {ranges} ({len(self.indices)} of {last + 1} usable)")
        elif (lo, hi) == (0, last):
            print(f"Frames {lo}..{hi} ({len(self.indices)}) - WHOLE VIDEO, no split "
                  f"held out. Pass --manifest to train on the train split only.")
        else:
            print(f"Frames {lo}..{hi} ({len(self.indices)} of {last + 1} usable)")

        self.crops = None
        if cache:
            self._build_cache(cache_threads)

    @staticmethod
    def _load_lms(lms_path):
        lms_list = []
        with open(lms_path, "r") as f:
            lines = f.read().splitlines()
            for line in lines:
                arr = line.split(" ")
                arr = np.array(arr, dtype=np.float32)
                lms_list.append(arr)
        return np.array(lms_list, dtype=np.int32)

    @staticmethod
    def _crop(img, lms):
        """The 320x320 face crop the model sees. Shared by the cached and the
        uncached path so both produce the same pixels by construction."""
        xmin = lms[1][0]
        ymin = lms[52][1]
        xmax = lms[31][0]
        width = xmax - xmin
        ymax = ymin + width
        crop_img = img[ymin:ymax, xmin:xmax]
        # NB the third positional argument of cv2.resize is dst, not the
        # interpolation, so this is INTER_LINEAR. Kept as-is: every trained
        # checkpoint saw exactly these pixels.
        crop_img = cv2.resize(crop_img, (328, 328), cv2.INTER_AREA)
        return crop_img[4:324, 4:324].copy()

    def _crop_frame(self, idx):
        return self._crop(cv2.imread(self.img_path_list[idx]),
                          self._load_lms(self.lms_path_list[idx]))

    def _build_cache(self, threads):
        import time
        t0 = time.time()
        self.crops = np.empty((len(self.indices), 320, 320, 3), dtype=np.uint8)
        self.row = {idx: r for r, idx in enumerate(self.indices)}
        # cv2 releases the GIL in imread/resize, so threads scale on the CPUs
        with ThreadPoolExecutor(max(1, threads)) as pool:
            for r, crop in enumerate(pool.map(self._crop_frame, self.indices)):
                self.crops[r] = crop
        print(f"Crop cache: {len(self.indices)} frames, "
              f"{self.crops.nbytes / 1e9:.2f} GB, built in {time.time() - t0:.0f}s")

    def __len__(self):
        return len(self.indices)
    
    def get_audio_features(self, features, index):
        left = index - 8
        right = index + 8
        pad_left = 0
        pad_right = 0
        if left < 0:
            pad_left = -left
            left = 0
        if right > features.shape[0]:
            pad_right = right - features.shape[0]
            right = features.shape[0]
        auds = torch.from_numpy(features[left:right])
        if pad_left > 0:
            auds = torch.cat([torch.zeros_like(auds[:pad_left]), auds], dim=0)
        if pad_right > 0:
            auds = torch.cat([auds, torch.zeros_like(auds[:pad_right])], dim=0) # [8, 16]
        return auds
    
    def get_audio_features_1(self, features, index):
    
        left = index - 8
        pad_left = 0
        if left < 0:
            pad_left = -left
            left = 0
        auds = features[left:index]
        auds = torch.from_numpy(auds)
        if pad_left > 0:
            # pad may be longer than auds, so do not use zeros_like
            auds = torch.cat([torch.zeros(pad_left, *auds.shape[1:], device=auds.device, dtype=auds.dtype), auds], dim=0)
        return auds
    
    def _tensors(self, img_real, img_real_ex):
        """Model inputs from two 320x320 crops; mirrors process_img exactly."""
        img_real_ori = img_real.copy()
        img_masked = apply_mouth_mask(img_real, self.mask_version)
        img_real_ori = img_real_ori.transpose(2,0,1).astype(np.float32)
        img_masked = img_masked.transpose(2,0,1).astype(np.float32)
        img_real_ex = img_real_ex.transpose(2,0,1).astype(np.float32)

        img_real_ex_T = torch.from_numpy(img_real_ex / 255.0)
        img_real_T = torch.from_numpy(img_real_ori / 255.0)
        img_masked_T = torch.from_numpy(img_masked / 255.0)
        img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=0)
        return img_concat_T, img_real_T

    def process_img(self, img, lms_path, img_ex, lms_path_ex):

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
        # ymax = ymin + width//7*6
        ymax = ymin + width
        # ymax = lms[16][1] + width//15
        
        crop_img = img[ymin:ymax, xmin:xmax]
        crop_img = cv2.resize(crop_img, (328, 328), cv2.INTER_AREA)
        img_real = crop_img[4:324, 4:324].copy()
        img_real_ori = img_real.copy()
        img_masked = apply_mouth_mask(img_real, self.mask_version)
        
        lms_list = []
        with open(lms_path_ex, "r") as f:
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
        # ymax = ymin + width//7*6
        ymax = ymin + width
        # ymax = lms[16][1] + width//15
        crop_img = img_ex[ymin:ymax, xmin:xmax]
        crop_img = cv2.resize(crop_img, (328, 328), cv2.INTER_AREA)
        img_real_ex = crop_img[4:324, 4:324].copy()
        
        img_real_ori = img_real_ori.transpose(2,0,1).astype(np.float32)
        img_masked = img_masked.transpose(2,0,1).astype(np.float32)
        img_real_ex = img_real_ex.transpose(2,0,1).astype(np.float32)
        
        img_real_ex_T = torch.from_numpy(img_real_ex / 255.0)
        img_real_T = torch.from_numpy(img_real_ori / 255.0)
        img_masked_T = torch.from_numpy(img_masked / 255.0)
        img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=0)

        return img_concat_T, img_real_T

    def __getitem__(self, i):
        if self.crops is not None:
            idx = self.indices[i]
            ex_int = self.indices[random.randint(0, len(self.indices)-1)]
            img_concat_T, img_real_T = self._tensors(self.crops[self.row[idx]],
                                                     self.crops[self.row[ex_int]])
            return img_concat_T, img_real_T, self._audio(idx)

        idx = self.indices[i]
        img = cv2.imread(self.img_path_list[idx])
        lms_path = self.lms_path_list[idx]

        # The appearance reference must come from the SAME split as the target.
        # Drawing it from the whole video is a second, quieter leak: the target
        # frame stays in-split while the reference carries held-out pixels in.
        ex_int = self.indices[random.randint(0, len(self.indices)-1)]
        img_ex = cv2.imread(self.img_path_list[ex_int])
        lms_path_ex = self.lms_path_list[ex_int]
        
        img_concat_T, img_real_T = self.process_img(img, lms_path, img_ex, lms_path_ex)
        return img_concat_T, img_real_T, self._audio(idx)

    def _audio(self, idx):
        audio_feat = self.get_audio_features(self.audio_feats, idx)
        
        if self.mode == "wenet":
            audio_feat = audio_feat.reshape(256,16,32)
        if self.mode == "hubert":
            audio_feat = audio_feat.reshape(32,32,32)
        if self.mode == "ave":
            audio_feat = audio_feat.reshape(32,16,16)
        if self.mode == "ssl":
            # 16 frames x 1024 dims = 16384 = 16*32*32
            audio_feat = audio_feat.reshape(16,32,32)
        return audio_feat
    
        