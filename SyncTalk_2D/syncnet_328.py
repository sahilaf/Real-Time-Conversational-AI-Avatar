import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import cv2
import os
import numpy as np
from torch import optim
from tqdm import tqdm
import random
import argparse



class Dataset(object):
    def __init__(self, dataset_dir, mode, split="all", val_ratio=0.1, neg_prob=0.5, min_neg_gap=5):

        self.img_path_list = []
        self.lms_path_list = []

        for i in range(len(os.listdir(dataset_dir+"/full_body_img/"))):

            img_path = os.path.join(dataset_dir+"/full_body_img/", str(i)+".jpg")
            lms_path = os.path.join(dataset_dir+"/landmarks/", str(i)+".lms")
            self.img_path_list.append(img_path)
            self.lms_path_list.append(lms_path)

        if mode=="wenet":
            audio_feats_path = dataset_dir+"/aud_wenet.npy"
        if mode=="hubert":
            audio_feats_path = dataset_dir+"/aud_hu.npy"
        if mode=="ave":
            audio_feats_path = dataset_dir+"/aud_ave.npy"
        if mode=="ssl":
            audio_feats_path = dataset_dir+"/aud_ssl.npy"
        self.mode = mode
        self.audio_feats = np.load(audio_feats_path)
        self.audio_feats = self.audio_feats.astype(np.float32)

        # Contiguous train/val split (neighboring frames are correlated,
        # so a random split would leak train info into val).
        n_total = min(self.audio_feats.shape[0] - 1, len(self.img_path_list))
        n_val = max(int(n_total * val_ratio), 32) if split != "all" else 0
        if split == "train":
            self.start, self.end = 0, n_total - n_val
        elif split == "val":
            self.start, self.end = n_total - n_val, n_total
        else:
            self.start, self.end = 0, n_total
        self.neg_prob = neg_prob
        self.min_neg_gap = min_neg_gap

    def __len__(self):

        return self.end - self.start

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
        ymax = ymin + width
        # ymax = lms[16][1] + width//15        
        # ymax = ymin + width//7*6
        crop_img = img[ymin:ymax, xmin:xmax]
        crop_img = cv2.resize(crop_img, (328, 328), cv2.INTER_AREA)
        img_real = crop_img[4:324, 4:324].copy()
        img_real_ori = img_real.copy()
        img_real_ori = img_real_ori.transpose(2,0,1).astype(np.float32)
        img_real_T = torch.from_numpy(img_real_ori / 255.0)
        
        return img_real_T

    def __getitem__(self, i):
        idx = self.start + i
        img = cv2.imread(self.img_path_list[idx])
        lms_path = self.lms_path_list[idx]

        ex_int = random.randint(self.start, self.end - 1)
        img_ex = cv2.imread(self.img_path_list[ex_int])
        lms_path_ex = self.lms_path_list[ex_int]

        img_real_T = self.process_img(img, lms_path, img_ex, lms_path_ex)

        # Contrastive sampling: 50% positive (matching audio), 50% negative
        # (audio from a temporally distant frame within the same split).
        if random.random() < self.neg_prob and (self.end - self.start) > 2 * self.min_neg_gap:
            wrong = random.randint(self.start, self.end - 1)
            while abs(wrong - idx) < self.min_neg_gap:
                wrong = random.randint(self.start, self.end - 1)
            audio_feat = self.get_audio_features(self.audio_feats, wrong)
            y = torch.zeros(1).float()
        else:
            audio_feat = self.get_audio_features(self.audio_feats, idx)
            y = torch.ones(1).float()

        if self.mode=="ave":
            audio_feat = audio_feat.reshape(32,16,16)
        elif self.mode=="ssl":
            # 16 frames x 1024 dims = 16384 = 16*32*32
            audio_feat = audio_feat.reshape(16,32,32)
        else:
            audio_feat = audio_feat.reshape(32,32,32)

        return img_real_T, audio_feat, y

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.residual = residual
    
    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class nonorm_Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)

class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)

class SyncNet_color(nn.Module):
    def __init__(self, mode):
        super(SyncNet_color, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv2d(3, 32, kernel_size=(7, 7), stride=1, padding=3),
            Conv2d(32, 32, kernel_size=5, stride=2, padding=1),


            Conv2d(32, 64, kernel_size=5, stride=2, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            )
            
        
        p1 = 128
        p2 = (1, 2)
        if mode == "hubert":
            p1 = 32
            p2 = (2, 2)
        if mode == "ave":
            p1 = 32
            p2 = 1
        if mode == "ssl":
            # [16,32,32] input - same spatial path as hubert, fewer channels.
            p1 = 16
            p2 = (2, 2)
        self.audio_encoder = nn.Sequential(
            Conv2d(p1, 128, kernel_size=3, stride=1, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(128, 256, kernel_size=3, stride=p2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 256, kernel_size=3, stride=2, padding=2),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

    def forward(self, face_sequences, audio_sequences): # audio_sequences := (B, dim, T)
        face_embedding = self.face_encoder(face_sequences) # (B, 512, 3, 3)
        audio_embedding = self.audio_encoder(audio_sequences)

        # reshape, not view: under channels_last the encoder output is not
        # contiguous and view() raises. reshape() is identical whenever view()
        # would have worked, and copies only when it would not.
        audio_embedding = audio_embedding.reshape(audio_embedding.size(0), -1)
        face_embedding = face_embedding.reshape(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)
        
        return audio_embedding, face_embedding

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    # Map cosine [-1, 1] -> probability (0, 1); raw cosine can be negative
    # with negative pairs, which BCELoss cannot accept.
    p = ((d + 1.0) * 0.5).clamp(1e-6, 1.0 - 1e-6)
    loss = logloss(p.unsqueeze(1), y)

    return loss

def evaluate(model, data_loader):
    model.eval()
    losses = []
    pos_sims, neg_sims = [], []
    with torch.no_grad():
        for imgT, audioT, y in tqdm(data_loader, desc="  validating", unit="batch", leave=False):
            imgT, audioT, y = imgT.cuda(), audioT.cuda(), y.cuda()
            a_emb, f_emb = model(imgT, audioT)
            losses.append(cosine_loss(a_emb, f_emb, y).item())
            d = nn.functional.cosine_similarity(a_emb, f_emb)
            pos_sims += d[y.squeeze(1) > 0.5].tolist()
            neg_sims += d[y.squeeze(1) <= 0.5].tolist()
    model.train()
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    return mean(losses), mean(pos_sims), mean(neg_sims)

def train(save_dir, dataset_dir, mode, epochs=100, batch_size=16, num_workers=4, lr=0.001,
          amp=False, resume="", init=""):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_dataset = Dataset(dataset_dir, mode=mode, split="train")
    val_dataset = Dataset(dataset_dir, mode=mode, split="val")
    # persistent_workers keeps the workers alive across epochs. Without it
    # Windows re-spawns a full Python+torch process per worker every epoch,
    # which is slow and spikes RAM.
    #
    # Validation gets half the workers: 0 on a 2-worker laptop (so we never
    # hold two sets of worker processes on 7.4 GB of RAM), but 4 when Colab
    # passes 8. Running validation single-threaded there cost ~11 s an epoch,
    # which is ~55 minutes across a 300-epoch run.
    val_workers = num_workers // 2
    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0))
    val_data_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=val_workers,
        persistent_workers=(val_workers > 0))
    model = SyncNet_color(mode).cuda()
    # --init warm-starts weights only (e.g. person fine-tune from the
    # universal Bangla SyncNet). --resume continues an interrupted run.
    if init:
        model.load_state_dict(torch.load(init, map_location="cuda"))
        print(f"Initialised weights from {init}.")
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=lr)

    use_amp = amp and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("AMP enabled (fp16 autocast). Validation stays fp32.")

    best_val_loss = float("inf")
    start_epoch = 0
    log_path = os.path.join(save_dir, "train_log.csv")
    if resume:
        ckpt = torch.load(resume, map_location="cuda")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and use_amp:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from {resume} at epoch {start_epoch}.")
    if not resume or not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_loss,val_pos_sim,val_neg_sim\n")
    for epoch in range(start_epoch, epochs):
        epoch_losses = []
        with tqdm(total=len(train_dataset), desc=f'Epoch {epoch+1}/{epochs}', unit='img') as p:
            for batch in train_data_loader:
                imgT, audioT, y = batch
                imgT = imgT.cuda()
                audioT = audioT.cuda()
                y = y.cuda()
                optimizer.zero_grad()
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    audio_embedding, face_embedding = model(imgT, audioT)
                # BCELoss is unsafe under autocast, so score in fp32.
                loss = cosine_loss(audio_embedding.float(), face_embedding.float(), y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_losses.append(loss.item())
                p.set_postfix(**{'loss': f"{loss.item():.4f}",
                                 'avg': f"{sum(epoch_losses)/len(epoch_losses):.4f}"})
                p.update(imgT.shape[0])
        train_loss = sum(epoch_losses) / len(epoch_losses)
        val_loss, val_pos, val_neg = evaluate(model, val_data_loader)
        print(f"epoch {epoch+1}  train {train_loss:.4f}  val {val_loss:.4f}  "
              f"pos_sim {val_pos:.4f}  neg_sim {val_neg:.4f}  gap {val_pos - val_neg:.4f}")
        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f},{val_pos:.6f},{val_neg:.6f}\n")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_val.pth"))
        if (epoch + 1) % 25 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, str(epoch+1)+'.pth'))
        # last.pth carries the full resume state. best_val.pth and the numbered
        # files stay plain state_dicts so eval_sync_328.py loads them unchanged.
        torch.save({"epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_val_loss": best_val_loss},
                   os.path.join(save_dir, "last.pth"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='test', type=str)
    parser.add_argument('--dataset_dir', default='./dataset/May', type=str)
    parser.add_argument('--asr', default='ave', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    # 2 is right for a 7.4 GB RAM laptop; raise to 8 on Colab.
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--amp', action='store_true',
                        help="Mixed precision. Big speedup on A100; changes numerics, so off by default.")
    parser.add_argument('--resume', default="", type=str,
                        help="Path to a last.pth to continue an interrupted run.")
    parser.add_argument('--init', default="", type=str,
                        help="Warm-start weights from a checkpoint (e.g. the universal Bangla SyncNet).")
    opt = parser.parse_args()

    train(opt.save_dir, opt.dataset_dir, opt.asr, opt.epochs, opt.batch_size, opt.num_workers, opt.lr,
          amp=opt.amp, resume=opt.resume, init=opt.init)