
import os
import librosa
import librosa.filters
from scipy import signal
from os.path import basename
import numpy as np


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F




class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, leakyReLU=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout)
        )
        if leakyReLU:
            self.act = nn.LeakyReLU(0.02)
        else:
            self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0), )

    def forward(self, x):
        out = self.audio_encoder(x)
        out = out.squeeze(2).squeeze(2)

        return out
    
# ---------------------------------------------------------------------------
# Mouth mask
# ---------------------------------------------------------------------------
# The generator receives the current frame with the mouth blacked out. The
# original mask stopped at row 310 of a 320-row crop, leaving the chin and jaw
# visible - and jaw position predicts mouth opening almost perfectly.
#
# Measured on redwan with audio and appearance reference both held constant,
# so the masked frame was the only varying input:
#     mask reveals nothing          -> mouth motion 0.00
#     mask reveals only rows 310-319 -> mouth motion 4.52
#     mask as originally written     -> mouth motion 4.98
# i.e. that 10-row jaw strip accounted for ~91% of the motion. The generator
# was inferring mouth shape from the jaw rather than from the audio, which is
# why a silent track still produced a talking avatar.
#
# v2 extends the mask to the bottom edge so the jaw is hidden.
#
# THE MASK MUST BE IDENTICAL IN TRAINING, INFERENCE AND EVALUATION. Training
# writes its version to train_config.json beside the checkpoint; checkpoints
# predating that file are treated as legacy so old models still run correctly.
MASK_V2 = "v2_no_jaw"
MASK_LEGACY = "legacy"


def apply_mouth_mask(img320, version=MASK_V2):
    """Return a copy of the 320x320 crop with the mouth region blacked out."""
    out = img320.copy()
    if version == MASK_LEGACY:
        out[5:310, 5:315] = 0      # leaves rows 310-319 (the jaw) visible
    else:
        out[5:, 5:315] = 0         # jaw hidden
    return out


def suppress_invented_chroma(pred_bgr, real_bgr, threshold=0.03):
    """Remove blue/cyan the generator invented that is absent from the real face.

    A sync loss trained at weight 0.03 measurably improves lip-sync, but leaves a
    persistent blue smear running from the lower lip down through the beard. The
    cause is the loss, not the model: L1 error on near-black pixels is tiny in
    absolute terms, so the darkest region of the crop is the cheapest place for
    the generator to hide a perturbation, and nothing else in the objective
    penalises inventing colour (the perceptual term is weighted 0.01).

    Only pixels that are blue-dominant in the prediction AND *not* blue-dominant
    in the source face are altered. That leaves the subject's blue shirt - which
    is genuinely in frame at the bottom of the crop - untouched, and it cannot
    leak ground truth because the correction caps a channel rather than copying
    anything across.

    Luminance is preserved, so the generated mouth shape survives intact.

    Args:
        pred_bgr: generated crop, uint8 BGR.
        real_bgr: the same crop from the source frame, uint8 BGR, same shape.
        threshold: how far blue must exceed green and red to count as invented.
    Returns:
        Corrected uint8 BGR array, and the fraction of pixels changed.
    """
    if pred_bgr.shape != real_bgr.shape:
        raise ValueError(f"shape mismatch: {pred_bgr.shape} vs {real_bgr.shape}")

    pred = pred_bgr.astype(np.float32) / 255.0
    real = real_bgr.astype(np.float32) / 255.0

    pred_excess = pred[..., 0] - np.maximum(pred[..., 1], pred[..., 2])
    real_excess = real[..., 0] - np.maximum(real[..., 1], real[..., 2])
    invented = (pred_excess > threshold) & (real_excess <= threshold)

    # Cap blue at the other channels outright. An absolute tolerance is
    # meaningless here: in the beard, green and red sit around 0.05, so allowing
    # blue to exceed them by even 0.10 leaves it three times brighter than both
    # and still plainly blue. Removing the dominance entirely is what the region
    # actually needs, and a face has no legitimately blue pixels anyway.
    out = pred.copy()
    ceiling = np.maximum(pred[..., 1], pred[..., 2])
    out[..., 0] = np.where(invented, ceiling, pred[..., 0])
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8), float(invented.mean())


def read_mask_version(checkpoint_path):
    """Mask version a checkpoint was trained with. Absent marker => legacy."""
    import json
    d = checkpoint_path if os.path.isdir(checkpoint_path) else os.path.dirname(checkpoint_path)
    p = os.path.join(d, "train_config.json")
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f).get("mask_version", MASK_LEGACY)
        except Exception:
            pass
    return MASK_LEGACY


def get_audio_features(features, index):
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


def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]


def preemphasis(wav, k):
    return signal.lfilter([1, -k], [1], wav)


def melspectrogram(wav):
    D = _stft(preemphasis(wav, 0.97))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - 20

    return _normalize(S)


def _stft(y):
    return librosa.stft(y=y, n_fft=800, hop_length=200, win_length=800)


def _linear_to_mel(spectogram):
    global _mel_basis
    _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectogram)


def _build_mel_basis():
    return librosa.filters.mel(sr=16000, n_fft=800, n_mels=80, fmin=55, fmax=7600)


def _amp_to_db(x):
    min_level = np.exp(-5 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _normalize(S):
    return np.clip((2 * 4.) * ((S - -100) / (--100)) - 4., -4., 4.)


class AudDataset(object):
    def __init__(self, wavpath):
        wav = load_wav(wavpath, 16000)

        self.orig_mel = melspectrogram(wav).T
        self.data_len = int((self.orig_mel.shape[0] - 16) / 80. * float(25)) + 2

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(25)))

        end_idx = start_idx + 16
        if end_idx > spec.shape[0]:
            # print(end_idx, spec.shape[0])
            end_idx = spec.shape[0]
            start_idx = end_idx - 16

        return spec[start_idx: end_idx, :]

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):

        mel = self.crop_audio_window(self.orig_mel.copy(), idx)
        if (mel.shape[0] != 16):
            raise Exception('mel.shape[0] != 16')
        mel = torch.FloatTensor(mel.T).unsqueeze(0)

        return mel
