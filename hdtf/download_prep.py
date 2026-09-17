"""Rebuild the frozen HDTF test subset from YouTube.

Downloads only the needed ~35s section of each source video rather than the
whole thing, applies HDTF's published crop window, and emits a 30.88s clip at
25fps / 512x512 plus 16kHz mono PCM audio - matching the redwan test clip so
frame counts line up across datasets.

Media is never redistributed; only this script and the manifest are.

  python download_prep.py            # all clips, skips ones already done
  python download_prep.py --only AdamSchiff
"""
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
MAN = json.load(open(os.path.join(HERE, 'hdtf_test_manifest.json')))
RAW, OUT = os.path.join(HERE, 'raw'), os.path.join(HERE, 'clips')
DUR = MAN['clip_seconds']          # 30.88
LEAD = 5                           # start this far into the annotated range
PAD = 3                            # extra tail so the cut never runs short


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def fetch(c):
    """Grab only the needed section. Returns path or None."""
    dst = os.path.join(RAW, f"{c['identity']}.mp4")
    if os.path.exists(dst) and os.path.getsize(dst) > 1e5:
        return dst
    a = c['start'] + LEAD
    b = a + DUR + PAD
    # height cap follows the resolution HDTF used, so the published crop window
    # lands where it was measured. Without it yt-dlp may hand back 4K or 480p
    # and the crop silently addresses the wrong pixels.
    hcap = c['resolution']
    cmd = (f'python -m yt_dlp -q --no-warnings '
           f'-f "bv*[height<={hcap}]+ba/b[height<={hcap}]" '
           f'--download-sections "*{a}-{b}" --force-keyframes-at-cuts '
           f'--merge-output-format mp4 -o "{dst}" "{c["url"]}"')
    r = sh(cmd)
    ok = os.path.exists(dst) and os.path.getsize(dst) > 1e5
    if not ok:
        err = (r.stderr or r.stdout).strip().split('\n')
        print(f"    FAILED: {err[-1][:150] if err else 'no output'}")
        return None
    return dst


def prepare(c, src):
    """Verify the source resolution, then crop -> 512x512 @25fps + 16k PCM."""
    ident = c['identity']
    probe = sh(f'ffprobe -v error -select_streams v:0 -show_entries '
               f'stream=width,height -of csv=p=0 "{src}"').stdout.strip()
    try:
        W, H = (int(v) for v in probe.split(',')[:2])
    except Exception:
        print(f'    probe failed: {probe!r}'); return None
    cr = c['crop']
    if cr['x'] + cr['w'] > W or cr['y'] + cr['h'] > H:
        # the crop was measured against a different resolution than we got
        print(f'    CROP OUT OF FRAME for {ident}: {cr} vs {W}x{H} - skipped')
        return None
    v = os.path.join(OUT, f'{ident}.mp4')
    a = os.path.join(OUT, f'{ident}.wav')
    sh(f'ffmpeg -y -loglevel error -i "{src}" -t {DUR} '
       f'-filter:v "crop={cr["w"]}:{cr["h"]}:{cr["x"]}:{cr["y"]},scale=512:512" '
       f'-r 25 -c:v libx264 -crf 12 -pix_fmt yuv420p -an "{v}"')
    sh(f'ffmpeg -y -loglevel error -i "{src}" -t {DUR} -vn -ac 1 -ar 16000 '
       f'-c:a pcm_s16le "{a}"')
    n = sh(f'ffprobe -v error -select_streams v:0 -count_frames '
           f'-show_entries stream=nb_read_frames -of csv=p=0 "{v}"').stdout.strip()
    return dict(identity=ident, frames=int(n) if n.isdigit() else 0,
                source_wh=[W, H], video=v, audio=a)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None)
    args = ap.parse_args()
    os.makedirs(RAW, exist_ok=True); os.makedirs(OUT, exist_ok=True)

    clips = [c for c in MAN['clips'] if not args.only or c['identity'] == args.only]
    done, failed = [], []
    for i, c in enumerate(clips, 1):
        ident = c['identity']
        print(f"[{i}/{len(clips)}] {ident}", flush=True)
        t0 = time.time()
        src = fetch(c)
        if not src:
            failed.append(dict(identity=ident, reason='download')); continue
        info = prepare(c, src)
        if not info:
            failed.append(dict(identity=ident, reason='crop/prepare')); continue
        print(f"    {info['frames']} frames from {info['source_wh']}  "
              f"({time.time()-t0:.0f}s)")
        done.append(info)

    rep = dict(attempted=len(clips), succeeded=len(done), failed=failed,
               retrieved=time.strftime('%Y-%m-%d'), clips=done)
    json.dump(rep, open(os.path.join(HERE, 'build_report.json'), 'w'), indent=1)
    print(f"\n{len(done)}/{len(clips)} rebuilt. attrition: {len(failed)}")
    for f in failed:
        print('  failed:', f['identity'], '-', f['reason'])
