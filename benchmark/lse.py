"""LSE-D / LSE-C via the ORIGINAL syncnet_python.

NOT Wav2Lip's evaluation/scores_LSE/SyncNetInstance_calc_scores.py, which
force-resizes the crop to 224x224 under a "#HARD CODED, CHANGE BEFORE RELEASE"
comment. The two implementations do not agree; say which one produced any
number you publish.

Audio is always muxed as PCM-in-AVI. AAC encoder priming shifts the apparent
AV offset by 2 frames (80 ms) - LSE-D/LSE-C themselves are read at the best
offset and barely move, but the reported offset column becomes wrong.
"""
import os
import re
import shutil
import subprocess
import tempfile

SYNC = os.environ.get('SYNCNET_DIR', '/content/syncnet_python')


def _sh(cmd, cwd=None):
    return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)


def mux_pcm(video, audio, out):
    """Attach audio as PCM. Never AAC."""
    _sh(f'ffmpeg -y -loglevel error -i "{video}" -i "{audio}" '
        f'-c:v copy -c:a pcm_s16le -map 0:v:0 -map 1:a:0 -shortest "{out}"')
    return os.path.exists(out) and os.path.getsize(out) > 1e4


def lse(video_with_audio, tag):
    """Returns (LSE_D, LSE_C, av_offset, error_text)."""
    work = tempfile.mkdtemp(prefix='syncwork_')
    try:
        p = _sh(f'python run_pipeline.py --videofile "{video_with_audio}" '
                f'--reference {tag} --data_dir {work}', cwd=SYNC)
        s = _sh(f'python run_syncnet.py --videofile "{video_with_audio}" '
                f'--reference {tag} --data_dir {work}', cwd=SYNC)
        txt = s.stdout + s.stderr
        d = re.search(r'Min dist:\s*([\d.]+)', txt)
        c = re.search(r'Confidence:\s*([\d.]+)', txt)
        o = re.search(r'AV offset:\s*(-?\d+)', txt)
        if not d:
            return None, None, None, (p.stderr + txt)[-600:]
        return (float(d.group(1)),
                float(c.group(1)) if c else None,
                int(o.group(1)) if o else None,
                '')
    finally:
        shutil.rmtree(work, ignore_errors=True)
