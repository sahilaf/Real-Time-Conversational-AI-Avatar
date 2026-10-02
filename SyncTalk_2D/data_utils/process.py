import os
import cv2
import json
import argparse
import multiprocessing as mp
import numpy as np
from tqdm import tqdm


def run(cmd, what):
    """Run a shell command and fail loudly. os.system returns a code that was
    previously ignored, so a failed ffmpeg produced a missing or truncated
    file and the error only surfaced much later as something confusing."""
    code = os.system(cmd)
    if code != 0:
        raise RuntimeError(f"{what} failed (exit {code}): {cmd}")


def extract_audio(path, out_path, sample_rate=16000, tempo=1.0):

    if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
        print(f'[INFO] {out_path} already present, skipping audio extraction')
        return
    print(f'[INFO] ===== extract audio from {path} to {out_path} =====')
    # -y so a re-run overwrites instead of hanging on ffmpeg's prompt, and
    # quoted paths so a directory with spaces does not split into arguments
    # tempo != 1 is --retime: the audio is sped up by exactly the factor the
    # frames are, so frame i and the audio that was spoken over it still meet.
    af = f'-af "atempo={tempo:.10f}" ' if abs(tempo - 1.0) > 1e-9 else ""
    run(f'ffmpeg -y -i "{path}" {af}-f wav -ar {sample_rate} "{out_path}"',
        "audio extraction")
    print(f'[INFO] ===== extracted audio =====')

def extract_images(path, retime=False):

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
    if (fps == 25 or retime) and expected > 0 and have >= expected:
        print(f"[INFO] {have} frames already extracted, skipping")
        return

    if fps != 25 and not retime:
        # High quality conversion to 25fps using ffmpeg
        converted = os.path.splitext(path)[0] + "_25fps.mp4"
        run(f'ffmpeg -y -i "{path}" -vf "fps=25" -c:v libx264 -c:a aac "{converted}"',
            "25fps conversion")
        path = converted

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps != 25 and not retime:
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


_worker_landmark = None


def _init_landmark_worker():
    """One detector per worker process. OpenCV's own thread pool is pinned to
    one thread so N processes do not each spawn N threads and thrash."""
    global _worker_landmark
    cv2.setNumThreads(1)
    from get_landmark import Landmark
    _worker_landmark = Landmark()


def _detect_chunk(job):
    """Detect faces for a list of frames; write each hit, return the misses."""
    full_img_dir, landmarks_dir, items = job
    missed = []
    for idx, img_name in items:
        result = _worker_landmark.detect(os.path.join(full_img_dir, img_name))
        if result is None:
            missed.append(idx)
            continue
        pre_landmark, x1, y1 = result
        write_lms(os.path.join(landmarks_dir, f"{idx}.lms"),
                  [(p[0] + x1, p[1] + y1) for p in pre_landmark])
    return missed


def get_landmark_parallel(path, landmarks_dir, workers):
    """get_landmark, with detection spread over `workers` processes.

    Detection is independent per frame, so it runs in parallel. The one
    order-dependent step - a frame with no face takes the landmarks of the
    nearest earlier frame that has one (or the first, for a leading run) -
    runs afterwards, sequentially, in timeline order, which is exactly what
    the serial version produces. Resume works the same way: frames with a
    valid .lms are skipped.
    """
    print(f"detecting landmarks with {workers} worker processes...")
    base = os.path.dirname(path)
    full_img_dir = os.path.join(base, "full_body_img")
    missing_log = os.path.join(base, "landmarks_missing.txt")
    frames = sorted(
        (f for f in os.listdir(full_img_dir) if f.endswith(".jpg")),
        key=lambda f: int(os.path.splitext(f)[0]),
    )
    todo = []
    for img_name in frames:
        idx = int(os.path.splitext(img_name)[0])
        if read_lms(os.path.join(landmarks_dir, f"{idx}.lms")) is None:
            todo.append((idx, img_name))
    print(f"[INFO] {len(frames) - len(todo)}/{len(frames)} landmarks already present")

    missing = set()
    if os.path.exists(missing_log):
        missing = {int(l) for l in open(missing_log).read().split() if l.strip()}

    if todo:
        chunk = 64
        jobs = [(full_img_dir, landmarks_dir, todo[i:i + chunk])
                for i in range(0, len(todo), chunk)]
        # spawn, not fork: each worker initialises CUDA for the landmark net
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_landmark_worker) as pool, \
                open(missing_log, "a") as log:
            with tqdm(total=len(todo)) as bar:
                for job, missed in zip(jobs, pool.imap(_detect_chunk, jobs)):
                    for idx in missed:
                        if idx not in missing:
                            missing.add(idx)
                            log.write(f"{idx}\n")
                    log.flush()
                    bar.update(len(job[2]))

    # Sequential fill, in timeline order - identical to the serial version.
    last_good, pending = None, []
    for img_name in frames:
        idx = int(os.path.splitext(img_name)[0])
        lms_path = os.path.join(landmarks_dir, f"{idx}.lms")
        existing = read_lms(lms_path) if idx not in missing else None
        if existing is not None:
            last_good = existing
            for p in pending:
                write_lms(p, existing)
            pending = []
            continue
        if last_good is None:
            pending.append(lms_path)
        else:
            write_lms(lms_path, last_good)
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
    print(f"[INFO] landmarks done; {len(ordered)} frames had no detectable face")


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
    parser.add_argument('--retime', action='store_true',
                        help="Video not at 25 fps: keep every native frame and treat it as "
                             "1/25 s, speeding the audio up by the same factor (25/fps). "
                             "Lossless; the default instead re-encodes with ffmpeg fps=25, "
                             "which duplicates or drops frames.")
    parser.add_argument('--workers', type=int, default=1,
                        help="Processes for landmark detection. 1 = the original serial path.")
    opt = parser.parse_args()

    base_dir = os.path.dirname(opt.path)
    wav_path = os.path.join(base_dir, 'aud.wav')
    landmarks_dir = os.path.join(base_dir, 'landmarks')

    os.makedirs(landmarks_dir, exist_ok=True)
    
    tempo = 1.0
    if opt.retime:
        _cap = cv2.VideoCapture(opt.path)
        native_fps = _cap.get(cv2.CAP_PROP_FPS)
        _cap.release()
        tempo = 25.0 / native_fps
        with open(os.path.join(base_dir, "retime.json"), "w") as f:
            json.dump({"native_fps": native_fps, "audio_tempo": tempo,
                       "note": "every native frame kept as 1/25 s; audio sped up by audio_tempo"},
                      f, indent=2)
        print(f"[INFO] retime: {native_fps} fps -> 25 fps, audio tempo x{tempo:.6f}")

    extract_audio(opt.path, wav_path, tempo=tempo)
    extract_images(opt.path, retime=opt.retime)
    if opt.workers > 1:
        get_landmark_parallel(opt.path, landmarks_dir, opt.workers)
    else:
        get_landmark(opt.path, landmarks_dir)
    get_audio_feature(wav_path)
    
    