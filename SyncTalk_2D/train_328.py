import argparse
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasetsss_328 import MyDataset
from syncnet_328 import SyncNet_color
from unet_328 import Model
import random
import torchvision.models as models

def get_args():
    parser = argparse.ArgumentParser(description='Train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--use_syncnet', action='store_true', help="if use syncnet, you need to set 'syncnet_checkpoint'")
    parser.add_argument('--syncnet_checkpoint', type=str, default="")
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--save_dir', type=str, help="trained model save path.")
    parser.add_argument('--see_res', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--asr', type=str, default="hubert")
    # 2 for a 7.4GB RAM laptop; raise to 8 on Colab. Was hardcoded 32.
    # This script is heavier than syncnet_328.py (VGG19 + SyncNet + UNet all
    # resident), and on Windows every worker is a fresh process that re-imports
    # torch, so 4 workers exhausted RAM during worker spawn. Use 0 if 2 fails.
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--amp', action='store_true',
                        help="Mixed precision. Big speedup on A100; changes numerics, so off by default.")
    parser.add_argument('--resume', type=str, default="",
                        help="Path to a last.pth to continue an interrupted run.")
    # Upstream hardcoded 10, which was harmless only because their SyncNet was
    # collapsed. With a working SyncNet, 10x makes the generator adversarially
    # maximise the frozen scorer and emit noise. Wav2Lip uses 0.03.
    parser.add_argument('--sync_weight', type=float, default=0.03,
                        help="Weight on the sync loss. 10 (the old hardcoded value) destroys "
                             "the image once the SyncNet actually works.")
    parser.add_argument('--sync_start_epoch', type=int, default=5,
                        help="Keep sync weight at 0 until this epoch so the generator learns "
                             "to reconstruct a face first.")
    parser.add_argument('--mask_version', type=str, default="v2_no_jaw",
                        choices=["v2_no_jaw", "legacy"],
                        help="v2_no_jaw hides the jaw so the model cannot infer mouth "
                             "shape from it. legacy reproduces the original leaky mask.")

    return parser.parse_args()

args = get_args()
use_syncnet = args.use_syncnet
# Loss functions
class PerceptualLoss():
    
    def contentFunc(self):
        conv_3_3_layer = 14
        # Slice on CPU and move only the part we keep. The original moved the
        # whole 20M-parameter VGG19 feature stack to the GPU and then discarded
        # everything past conv3_3, wasting both VRAM and host RAM on a machine
        # that has little of either.
        cnn = models.vgg19(pretrained=True).features
        model = nn.Sequential()
        for i,layer in enumerate(list(cnn)):
            model.add_module(str(i),layer)
            if i == conv_3_3_layer:
                break
        del cnn
        model = model.cuda()
        # VGG is a fixed feature extractor - nothing ever trains it, so the
        # gradient buffers it would otherwise allocate are pure waste.
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    def __init__(self, loss):
        self.criterion = loss
        self.contentFunc = self.contentFunc()

    def get_loss(self, fakeIm, realIm):
        f_fake = self.contentFunc.forward(fakeIm)
        # The real branch is detached anyway, so building an autograd graph for
        # it just retains ~700MB of activations until backward. Skip it.
        with torch.no_grad():
            f_real_no_grad = self.contentFunc.forward(realIm)
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    # Map cosine [-1, 1] -> probability (0, 1); a discriminative SyncNet can
    # emit negative cosines, which BCELoss cannot accept.
    p = ((d + 1.0) * 0.5).clamp(1e-6, 1.0 - 1e-6)
    loss = logloss(p.unsqueeze(1), y)

    return loss

def train(net, epoch, batch_size, lr):
    content_loss = PerceptualLoss(torch.nn.MSELoss())
    if use_syncnet:
        if args.syncnet_checkpoint == "":
            raise ValueError("Using syncnet, you need to set 'syncnet_checkpoint'.Please check README")
            
        syncnet = SyncNet_color(args.asr).eval().cuda()
        syncnet.load_state_dict(torch.load(args.syncnet_checkpoint))
    save_dir= args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Record how this model was trained so inference/eval reproduce it exactly.
    # A checkpoint dir without this file is assumed legacy (pre-jaw-mask-fix).
    import json as _json
    with open(os.path.join(save_dir, "train_config.json"), "w") as _f:
        _json.dump({"mask_version": args.mask_version, "asr": args.asr,
                    "use_syncnet": bool(use_syncnet),
                    "syncnet_checkpoint": args.syncnet_checkpoint,
                    "sync_weight": args.sync_weight,
                    "sync_start_epoch": args.sync_start_epoch}, _f, indent=2)
    print(f"Sync loss: weight {args.sync_weight} from epoch {args.sync_start_epoch}")
    print(f"Mouth mask: {args.mask_version} "
          f"({'jaw hidden' if args.mask_version != 'legacy' else 'jaw VISIBLE - leaks mouth shape'})")
    dataloader_list = []
    dataset_list = []
    dataset_dir_list = [args.dataset_dir]
    for dataset_dir in dataset_dir_list:
        dataset = MyDataset(dataset_dir, args.asr, mask_version=args.mask_version)
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                      drop_last=False, num_workers=args.num_workers,
                                      persistent_workers=(args.num_workers > 0))
        dataloader_list.append(train_dataloader)
        dataset_list.append(dataset)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.L1Loss()

    use_amp = args.amp and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("AMP enabled (fp16 autocast).")

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cuda")
        net.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and use_amp:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"]
        print(f"Resumed from {args.resume} at epoch {start_epoch}.")

    loss_log = os.path.join(save_dir, "loss_log.csv")
    if not args.resume or not os.path.exists(loss_log):
        with open(loss_log, "w") as f:
            f.write("epoch,l1,vgg,sync,sync_weight" + chr(10))

    for e in range(start_epoch, epoch):
        net.train()
        epoch_terms = []
        random_i = random.randint(0, len(dataset_dir_list)-1)
        dataset = dataset_list[random_i]
        train_dataloader = dataloader_list[random_i]
        
        with tqdm(total=len(dataset), desc=f'Epoch {e + 1}/{epoch}', unit='img') as p:
            for batch in train_dataloader:
                imgs, labels, audio_feat = batch
                imgs = imgs.cuda()
                labels = labels.cuda()
                audio_feat = audio_feat.cuda()
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    preds = net(imgs, audio_feat)
                    if use_syncnet:
                        a, v = syncnet(preds, audio_feat)
                    loss_PerceptualLoss = content_loss.get_loss(preds, labels)
                    loss_pixel = criterion(preds, labels)
                # BCELoss (inside cosine_loss) is unsafe under autocast, so the
                # sync term and the final sum are computed in fp32.
                # Sync weight is 0 until sync_start_epoch, so the generator first
                # learns to reconstruct a face. Handing a frozen scorer a large
                # weight from step 0 makes the generator maximise that scorer
                # adversarially instead of learning lip-sync - it produces
                # high-frequency colour noise that scores ~0.8 while the image
                # collapses. Upstream's 10x was harmless only because their
                # SyncNet was collapsed and contributed no gradient.
                w_sync = args.sync_weight if e >= args.sync_start_epoch else 0.0
                if use_syncnet and w_sync > 0:
                    y = torch.ones([preds.shape[0],1]).float().cuda()
                    sync_loss = cosine_loss(a.float(), v.float(), y)
                    loss = loss_pixel.float() + loss_PerceptualLoss.float()*0.01 + w_sync*sync_loss
                else:
                    sync_loss = torch.zeros((), device=preds.device)
                    loss = loss_pixel.float() + loss_PerceptualLoss.float()*0.01
                # Log the terms separately - a single total hides the generator
                # trading image quality away for sync score.
                p.set_postfix(**{'L1': f'{loss_pixel.item():.4f}',
                                 'vgg': f'{loss_PerceptualLoss.item():.3f}',
                                 'sync': f'{float(sync_loss):.4f}',
                                 'w': w_sync})
                epoch_terms.append((loss_pixel.item(), float(loss_PerceptualLoss),
                                    float(sync_loss)))
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                p.update(imgs.shape[0])
                
        if epoch_terms:
            import numpy as _np
            m = _np.mean(epoch_terms, axis=0)
            w_now = args.sync_weight if e >= args.sync_start_epoch else 0.0
            with open(loss_log, "a") as f:
                f.write(f"{e+1},{m[0]:.6f},{m[1]:.6f},{m[2]:.6f},{w_now}" + chr(10))
            print(f"epoch {e+1}  L1 {m[0]:.4f}  vgg {m[1]:.3f}  sync {m[2]:.4f}  (w={w_now})")

        # last.pth is the full resume state. The numbered files stay plain
        # state_dicts so inference/eval scripts keep loading them unchanged.
        torch.save({"epoch": e + 1,
                    "model": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict()},
                   os.path.join(save_dir, "last.pth"))
        if (e+1) % 5 == 0:
            torch.save(net.state_dict(), os.path.join(save_dir, str(e)+'.pth'))
        if args.see_res:
            net.eval()
            img_concat_T, img_real_T, audio_feat = dataset.__getitem__(random.randint(0, dataset.__len__()))
            img_concat_T = img_concat_T[None].cuda()
            audio_feat = audio_feat[None].cuda()
            with torch.no_grad():
                pred = net(img_concat_T, audio_feat)[0]
            pred = pred.cpu().numpy().transpose(1,2,0)*255
            pred = np.array(pred, dtype=np.uint8)
            img_real = img_real_T.numpy().transpose(1,2,0)*255
            img_real = np.array(img_real, dtype=np.uint8)
            cv2.imwrite("./train_tmp_img/epoch_"+str(e)+".jpg", pred)
            cv2.imwrite("./train_tmp_img/epoch_"+str(e)+"_real.jpg", img_real)
        
            

if __name__ == '__main__':
    
    
    net = Model(6, mode=args.asr).cuda()
    train(net, args.epochs, args.batchsize, args.lr)