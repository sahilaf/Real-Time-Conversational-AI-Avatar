"""Label every video frame of a Bangla clip with the mouth-shape group being spoken.

A Bangla wav2vec2 CTC model marks, every 20 ms, which Bangla character is being
produced. Each character maps to a group from Table 3.1 of the report, and each
40 ms video frame (25 fps) takes the group of the character nearest in time.
Frames with no speech energy are labelled silence.

Bangla spelling is close to phonemic, so characters stand in for phonemes. The
known approximation: a consonant written without a vowel sign carries an
inherent vowel (usually /ɔ/ or /o/) that this labelling files under the consonant.

CTC marks each character at a single frame ("spike") somewhere inside the sound,
not at its start, so the timing rule and any constant lag are calibrated once on
real video of a speaker (calibrate_timing in mouth_by_phoneme.py) and then applied
unchanged to every model output.

  python bangla_phonemes.py clip.wav out.npz        # needs torch >= 2.6
"""
import sys

import numpy as np
import soundfile as sf
import torch

MODEL = 'arijitx/wav2vec2-xls-r-300m-bengali'
FPS = 25

GROUPS = {
    'closed':  'পফবভম',                                  # V1 lips pressed together
    'open':    'আাঅ',                                     # V7, V8 jaw dropped
    'spread':  'ইঈিীএেয়',                                # V5, V6 lips spread
    'rounded': 'উঊুূওোঔৌঐৈ',                             # V9, V10 lips rounded
    'other':   'তথদধটঠডঢড়ঢ়নণলরসৎঞচছজঝযশষকখগঘঙংঃহঋৃ',     # V2-V4 lips neutral
}
CHAR2GROUP = {ch: g for g, chars in GROUPS.items() for ch in chars}
ORDER = ['silence', 'closed', 'open', 'spread', 'rounded', 'other']


def extract(wav_path, device='cpu'):
    """Run the CTC model once; return character spikes and per-frame energy."""
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    audio, sr = sf.read(wav_path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(1)
    assert sr == 16000, f'expected 16 kHz, got {sr}'
    proc = Wav2Vec2Processor.from_pretrained(MODEL)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL).to(device).eval()
    with torch.no_grad():
        x = proc(audio, sampling_rate=16000, return_tensors='pt').input_values.to(device)
        ids = model(x).logits[0].argmax(-1).cpu().numpy()
    vocab = {v: k for k, v in proc.tokenizer.get_vocab().items()}
    blank = proc.tokenizer.pad_token_id
    step = len(audio) / 16000 / len(ids)

    spike_t, spike_ch, prev = [], [], None
    for k, i in enumerate(ids):
        if i != blank and i != prev and vocab[i] not in ('|', ' '):
            spike_t.append((k + 0.5) * step)
            spike_ch.append(vocab[i])
        prev = i
    toks, prev = [], None
    for i in ids:
        if i != prev and i != blank:
            toks.append(vocab[i])
        prev = i

    n_v = int(round(len(audio) / 16000 * FPS))
    hop = 16000 // FPS
    rms = np.array([np.sqrt(np.mean(audio[j*hop:(j+1)*hop] ** 2) + 1e-12) for j in range(n_v)])
    return dict(spike_t=np.array(spike_t), spike_ch=np.array(spike_ch), rms=rms,
                transcript=''.join(toks).replace('|', ' '))


def assign(ex, n_frames, mode='nearest', offset=0):
    """Group per video frame. offset > 0 labels frame j with the sound at j - offset
    (i.e. the mouth lags the audio by `offset` frames)."""
    t, ch, rms = ex['spike_t'], ex['spike_ch'], ex['rms']
    silent = rms < 0.05 * np.percentile(rms, 95)
    out = []
    for j in range(n_frames):
        js = min(max(j - offset, 0), len(rms) - 1)
        if silent[js] or len(t) == 0:
            out.append('silence')
            continue
        tc = (js + 0.5) / FPS
        if mode == 'nearest':
            k = int(np.argmin(np.abs(t - tc)))
        else:  # 'hold': most recent spike at or before the frame centre
            k = max(int(np.searchsorted(t, tc, side='right')) - 1, 0)
        out.append(CHAR2GROUP.get(str(ch[k]), 'other'))
    return np.array(out)


if __name__ == '__main__':
    ex = extract(sys.argv[1])
    np.savez(sys.argv[2], **ex)
    print('TRANSCRIPT:', ex['transcript'])
    print('characters:', len(ex['spike_t']), '| frames:', len(ex['rms']))
