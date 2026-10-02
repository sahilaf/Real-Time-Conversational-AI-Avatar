"""Build colab/alapon_code.zip: the code and small model files that
Train_Alapon_v2.ipynb needs, and nothing else (no datasets, no checkpoints).

    python colab/make_code_bundle.py

Upload the zip to the Drive folder the notebook uses. Rebuild and re-upload it
whenever training or preprocessing code changes.
"""
import os
import zipfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "SyncTalk_2D")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alapon_code.zip")

FILES = [
    "train_328.py", "datasetsss_328.py", "unet_328.py", "syncnet_328.py", "utils.py",
    "data_utils/process.py", "data_utils/get_landmark.py", "data_utils/detect_face.py",
    "data_utils/pfld_mobileone.py", "data_utils/base_module.py", "data_utils/mean_face.txt",
    "data_utils/scrfd_2.5g_kps.onnx", "data_utils/checkpoint_epoch_335.pth.tar",
    "data_utils/ave/audio.py", "data_utils/ave/hparams.py", "data_utils/ave/test_w2l_audio.py",
    "data_utils/ave/checkpoints/audio_encoder.pth",
    "evaluation/scripts/create_holdout_manifest.py",
]

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for rel in FILES:
        src = os.path.join(ROOT, rel)
        if not os.path.exists(src):
            raise SystemExit(f"missing: SyncTalk_2D/{rel}")
        z.write(src, os.path.join("SyncTalk_2D", rel))

print(f"wrote {os.path.abspath(OUT)} ({os.path.getsize(OUT) / 1e6:.1f} MB, {len(FILES)} files)")
