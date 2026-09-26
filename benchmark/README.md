# Cross-system benchmark scorer

The metrics used to compare every lip-sync system on the same footing:
**LSE-D / LSE-C**, **LipLeak**, and **articulation**.

This code exists as files because the first version did not. It lived in Colab
cells, the session died, and it had to be rebuilt from a README. Anything that
produces a number in a paper belongs in the repository.

## Validation - do this before trusting any new number

The rebuild was checked against the 2026-09-13 redwan run, which is the only
reason it can be trusted. Re-run these whenever the scorer changes:

| check | rebuilt | 2026-09-13 | delta |
|---|---:|---:|---:|
| ground truth LSE-D | 7.396 | 7.436 | -0.040 |
| ground truth LSE-C | 5.136 | 5.103 | +0.033 |
| ground truth AV offset | 0 | 0 | exact |
| MuseTalk articulation | 0.269 | 0.269 | exact |
| Wav2Lip articulation | 0.610 | 0.621 | -0.011 |
| source articulation | 0.504 | 0.487 | +0.017 |

A scorer that drifts silently looks exactly like a dataset effect. Without this
table there is no way to tell the two apart.

## Reading the deltas

The articulation misses go in **opposite directions** and MuseTalk lands exactly
- that pattern is threshold noise, not a wrong landmark index. A wrong index or
face-width would bias every system the same way.

`open_fraction` thresholds at 0.03, which is the **median** of the source
aperture distribution, so the largest number of frames sit closest to the
boundary and landmark noise flips them. Reproducibility is about +/-0.02.
`median_aperture` is reported alongside for that reason and is stable.

## Alignment - verify per system, never assume

Every system truncates or pads differently, and a shifted output scores as a
merely mediocre system rather than an obviously broken one. Measured by sliding
the output against the source and comparing the **top 100 rows** only - above
the mouth, so untouched by any of these systems - then taking the minimum.

Sweep symmetrically (-6..+8). A minimum sitting at 0 in a one-sided sweep is
also what a clipped negative optimum looks like.

| system | offset | frames | correction |
|---|---:|---:|---|
| wav2lip / wav2lip_gan | 0 | 772 | none |
| musetalk_v1 / v15 | 0 | 772 | none |
| ip_lap | 0 | 768 | loses 4 at the TAIL; truncate source to match |
| latentsync | 0 | 772 | see below |

### LatentSync pads differently depending on how it is run

Verified 2026-09-18: run **unchunked** at 512x512, raw output is 774 frames and
**both head and tail align at offset 0** - so the 2 extra frames are at the
**END**.

The 2026-09-13 redwan run recorded the opposite ("prepends 2 frames per clip -
drop the FIRST two"), and it was right: that run was **chunked**, because
LatentSync OOMs on system RAM at 1080x1080 on a 13.6 GB box. The padding is a
per-chunk artefact.

Carrying the comment across configurations put a 2-frame shift into the first
HDTF run. Re-measure the padding whenever the chunking changes; do not trust a
previous run's note.


## Two traps that only appear when rescoring old archives

### Never re-mux a file that already has audio

Several redwan AVIs declare `avg_frame_rate 50/1` over 772 real frames.
`run_pipeline.py` re-extracts at `-r 25 -async 1`, and passing such a container
through ffmpeg again changes what it gets: ground truth moved **5.135 -> 6.109**
on LSE-C with byte-identical audio and `-c:v copy` video.

Files declaring `25/1` were unaffected. So the error hit six systems and spared
two - which is exactly why it survived a spot check. `score_redwan.py` probes for
an audio stream and muxes only when there is none.

The HDTF outputs genuinely carry no audio, so muxing is correct there. The rule
is not "always mux" or "never mux"; it is **do not transform a file that is
already in measurement-ready form**.

### Check the archive, not just the pipeline

`videos_redwan/ip_lap.avi` was stored UNCROPPED at 2160x1080 - IP_LAP's
`[sketch|result]` side-by-side, left half 91% black. Scored as found it reported
articulation **0.9935**; right-half cropped it reports **0.3854**.

The batch script cropped correctly. The archived copy never had been. A number
that absurd is easy to catch; one that is merely plausible is not.

## Layout assumed

    /content/repo/SyncTalk_2D     the landmarker (SYNCTALK_REPO)
    /content/syncnet_python       LSE (SYNCNET_DIR)
    /content/hdtf/clips           {id}.mp4, {id}.wav, {id}_silence.wav
    <drive>/.../hdtf              {id}__{system}__{real|silence}.avi

All overridable by the environment variables named in the source.

## Running

    python score_hdtf.py          # 78 jobs, 6 workers, ~30 min

**Resumable.** Each row is appended and fsynced as it lands, and a rerun skips
every (identity, system, cond) already in the CSV. Re-running after a crash
costs only the jobs that had not finished.

This is not decoration. The first version accumulated rows in memory and wrote
once at the end; a Colab runtime was reclaimed at 29/78 and took about twenty
minutes of compute with it. The CSV also defaults to Drive rather than
`/content` for the same reason.

**Pick the runtime by vCPU count, not by GPU.** The bottleneck is the SCRFD face
detector, which runs on CPU. Measured single-worker rates:

    Colab T4  ( 2 vCPU)   2.6 frames/s  ->  78 jobs in 4-5 hours
    Colab L4  (12 vCPU)   8.8 frames/s  ->  78 jobs in ~45 min
    Colab A100(12 vCPU)   7.9 frames/s

VRAM is irrelevant here - the landmarker and SyncNet are both small. A T4 looks
like the cheap choice and is nine times slower, which also means nine times the
exposure to having the runtime reclaimed mid-run.

`silence` rows get LipLeak only - there is nothing to be in sync with, so LSE is
not computed. `real` rows get LSE plus articulation. Ground truth is scored as a
row rather than assumed: real video is roughly LSE-D 7.4 / LSE-C 5.1, and
without that anchor Wav2Lip's LSE-C reads as better than a human.
