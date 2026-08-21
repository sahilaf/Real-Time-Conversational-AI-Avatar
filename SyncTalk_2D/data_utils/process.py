import os
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm


def run(cmd, what):
    """Run a shell command and fail loudly. os.system returns a code that was
    previously ignored, so a failed ffmpeg produced a missing or truncated
    file and the error only surfaced much later as something confusing."""
    code = os.system(cmd)
    if code != 0:
        raise RuntimeError(f"{what} failed (exit {code}): {cmd}")


def extract_audio(path, out_path, sample_rate=16000):

    if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
        print(f'[INFO] {out_path} already present, skipping audio extraction')
        return
    print(f'[INFO] ===== extract audio from {path} to {out_path} =====')
    # -y so a re-run overwrites instead of hanging on ffmpeg's prompt, and
    # quoted paths so a directory with spaces does not split into arguments
    run(f'ffmpeg -y -i "{path}" -f wav -ar {sample_rate} "{out_path}"',
        "audio extraction")
    print(f'[INFO] ===== extracted audio =====')

def extract_images(path):

    # os.path, not path.split("/"): on Windows the path separator is a
    # backslash, so the old string surgery silently produced a wrong directory
    full_body_dir = os.path.join(os.path.dirname(path), "full_body_img")
    os.makedirs(full_body_dir, exist_ok=True)

    counter = 0
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Resume: frames already on disk are worth minutes on a long batch. Only
    # skip when the count looks complete, so a half-extracted directory is
    # redone rather than silently accepted.
    have = len([f for f in os.listdir(full_body_dir) if f.endswith(".jpg")])
    if fps == 25 and expected > 0 and have >= expected:
        print(f"[INFO] {have} frames already extracted, skipping")
        return

    if fps != 25:
        # High quality conversion to 25fps using ffmpeg
        converted = os.path.splitext(path)[0] + "_25fps.mp4"
        run(f'ffmpeg -y -i "{path}" -vf "fps=25" -c:v libx264 -c:a aac "{converted}"',
            "25fps conversion")
        path = converted

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps != 25:
        raise ValueError("Your video fps should be 25!!!")

    print("extracting images...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(full_body_dir, f"{counter}.jpg"), frame)
        counter += 1
    cap.release()
    print(f"[INFO] extracted {counter} frames")
        
def get_audio_feature(wav_path):
    
    print("extracting audio feature...")
    run(f'python ./data_utils/ave/test_w2l_audio.py --wav_path "{wav_path}"',
        "audio feature extraction")
    
def write_lms(lms_path, pts):
    """Write absolute landmark coordinates, atomically.

    Via a temp file so an interrupted run cannot leave a half-written .lms
    that a resume would mistake for finished work.
    """
    tmp = lms_path + ".part"
    with open(tmp, "w") as f:
        for x, y in pts:
            f.write(str(x))
            f.write(" ")
            f.write(str(y))
            f.write("\n")
    os.replace(tmp, lms_path)


def read_lms(lms_path):
    """Absolute coords from an existing .lms, or None if unusable."""
    try:
        pts = [tuple(float(v) for v in line.split()) for line in
               open(lms_path).read().splitlines() if line.strip()]
        return pts if len(pts) >= 100 else None
    except Exception:
        return None


def get_landmark(path, landmarks_dir):
    """Landmarks for every frame, with no gaps and no renumbering.

    A frame with no detectable face used to abort the whole run. It must not
    simply be skipped either: downstream code reads full_body_img/{i}.jpg
    alongside landmarks/{i}.lms and treats frame i as audio time i/25, so a
    missing .lms breaks indexing and a *removed* frame silently shifts every
    later frame against the audio - which is fatal for a lip-sync dataset.

    So the frame is kept and its landmarks are carried over from the nearest
    frame that did resolve. Those frames are recorded in landmarks_missing.json
    so the clip filter can exclude any segment that leans on filled-in data.
    """
    print("detecting landmarks...")
    base = os.path.dirname(path)
    full_img_dir = os.path.join(base, "full_body_img")
    missing_log = os.path.join(base, "landmarks_missing.txt")

    # numeric order, not lexicographic: carrying forward only makes sense
    # along the real timeline (otherwise 10.jpg follows 1.jpg)
    frames = sorted(
        (f for f in os.listdir(full_img_dir) if f.endswith(".jpg")),
        key=lambda f: int(os.path.splitext(f)[0]),
    )

    # Resume support. Detection runs at ~7 fps and dominates the runtime, so
    # a batch spanning hours must not restart a video from zero after an
    # interruption. Frames whose .lms is already valid are skipped; the
    # missing-frame log is appended to as we go so it survives too.
    missing = set()
    if os.path.exists(missing_log):
        missing = {int(l) for l in open(missing_log).read().split() if l.strip()}
    done = {int(os.path.splitext(f)[0]) for f in os.listdir(landmarks_dir)
            if f.endswith(".lms")}
    if done:
        print(f"[INFO] resuming: {len(done)}/{len(frames)} landmarks already present")

    from get_landmark import Landmark
    landmark = Landmark()

    last_good = None            # absolute coords of the most recent detection
    pending = []                # leading frames seen before any detection
    log = open(missing_log, "a")

    for img_name in tqdm(frames):
        idx = int(os.path.splitext(img_name)[0])
        lms_path = os.path.join(landmarks_dir, f"{idx}.lms")

        if idx in done:
            existing = read_lms(lms_path)
            if existing is not None:
                last_good = existing        # keep carry-forward continuous
                continue                    # else fall through and redo it

        result = landmark.detect(os.path.join(full_img_dir, img_name))
        if result is None:
            if idx not in missing:
                missing.add(idx)
                log.write(f"{idx}\n")
                log.flush()
            if last_good is None:
                # nothing to copy yet; fill these once the first face appears
                pending.append(lms_path)
                continue
            write_lms(lms_path, last_good)
            continue

        pre_landmark, x1, y1 = result
        pts = [(p[0] + x1, p[1] + y1) for p in pre_landmark]
        last_good = pts
        write_lms(lms_path, pts)
        for p in pending:       # backfill the leading run
            write_lms(p, pts)
        pending = []

    log.close()
    if pending:
        raise RuntimeError(
            f"no face detected in ANY of {len(frames)} frames of {path} - "
            "wrong video, or the detector cannot see this footage")

    ordered = sorted(missing)
    report = {
        "video": os.path.basename(path),
        "frames": len(frames),
        "missing_count": len(ordered),
        "missing_pct": round(len(ordered) / max(len(frames), 1) * 100, 3),
        "missing_frames": ordered,
    }
    with open(os.path.join(base, "landmarks_missing.json"), "w") as f:
        json.dump(report, f, indent=2)

    if missing:
        print(f"[WARN] {len(missing)}/{len(frames)} frames "
              f"({report['missing_pct']}%) had no detectable face; landmarks "
              f"carried over from neighbours. See landmarks_missing.json")
    else:
        print(f"[INFO] all {len(frames)} frames resolved a face")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to video file")
    opt = parser.parse_args()

    base_dir = os.path.dirname(opt.path)
    wav_path = os.path.join(base_dir, 'aud.wav')
    landmarks_dir = os.path.join(base_dir, 'landmarks')

    os.makedirs(landmarks_dir, exist_ok=True)
    
    extract_audio(opt.path, wav_path)
    extract_images(opt.path)
    get_landmark(opt.path, landmarks_dir)
    get_audio_feature(wav_path)
    
    