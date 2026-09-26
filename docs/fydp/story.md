# Alapon — the first real-time Bangla AI avatar

**Pitch narrative for the FYDP defence.**

---

## The opening

> **"240 million people. Zero products."**
>
> "Every AI avatar on the market is English-first. We tested the six leading
> systems in the world. Not one of them has ever seen Bangla.
>
> We built the one that has — and it runs on this laptop."

Then connect, and let someone talk to it.

**Don't narrate the demo.** Let it play for thirty seconds. The silence while a
machine answers a question in fluent Bangla does more than any slide.

---

## The moment that wins the room

Everyone compares lip-sync systems by who scores highest.

**We added one row to the table that nobody adds: real human video.**

| | lip-sync score | |
|---|---:|---|
| **A real person, actually speaking** | **5.135** | *the target* |
| **Alapon (ours)** | **5.111** | **0.5% away** ✅ |
| LatentSync 1.5 (5 GB) | 5.088 | 0.9% away |
| MuseTalk v1.5 (3.4 GB) | 4.899 | 4.6% away |
| IP-LAP | 4.361 | 15% away |
| **Wav2Lip** | **6.444** | **25% BETTER than a human** ⚠️ |

Wav2Lip — one of the most-cited systems in the field — **scores 25% better at
lip-sync than a real human being speaking with their own mouth.**

That is not possible. A real person is perfectly in sync with their own voice by
definition. It is the ceiling.

> **"That's not excellence. That's a broken ruler — and almost nobody in this field
> has noticed, because almost nobody scores real video."**

Once you add that row, the leaderboard stops making sense. And the right question
stops being *who scores highest* and becomes **who lands closest to real.**

**We land closest.**

---

## What it costs them to get there

The honest read: LatentSync matches our quality. Here is the bill.

| | **Alapon** | LatentSync 1.5 | MuseTalk v1.5 | Wav2Lip |
|---|---:|---:|---:|---:|
| **Model size** | **49 MB** | 5,072 MB | 3,400 MB | 436 MB |
| | **1×** | **104× larger** | 69× | 9× |
| **Distance from real** | **0.5%** | 0.9% | 4.6% | *above real* |
| **Mouth movement vs real** | **0.94** | 0.79 | 0.53 | 1.05 |
| **Copied, not generated** | **0.001** | 0.359 | 0.004 | 0.458 |
| **Runs on a 4 GB laptop GPU** | **Yes** | No | No | — |

Two rows deserve a sentence each.

**Mouth movement (1.00 = like a real person).** MuseTalk scores respectably while
opening its mouth **half as much as a human**. It avoids mistakes by barely moving.
Alapon moves **94% as much as real video**.

**Copied, not generated.** Feed the model pure digital silence alongside a real
video. Anything the mouth does now is copied from the input — not generated from
audio. LatentSync: **0.359**. Alapon: **0.001**.

> **"Three hundred and fifty times less. Their quality is partly borrowed. Ours
> isn't."**

---

## It runs here

Not on a cluster. On a laptop, measured on the machine in this room.

| | **RTX 3050 Laptop, 4 GB** |
|---|---|
| Render speed | **30 frames per second** |
| Live system, end to end | **~28 fps sustained** |
| What the format needs | 25 fps — **cleared, with headroom** |
| Video memory used | **0.30 GB of 4 GB** |
| Model on disk | **49.2 MB** |

**0.3 gigabytes.** It doesn't need a dedicated GPU. It needs a corner of one.

> **"Same quality as a five-gigabyte model. One percent of the size. Running on a
> laptop, right now, in front of you."**

---

## Why that is the whole business

Size is not a spec. It decides where the product can exist.

| | **49 MB (Alapon)** | **5 GB (theirs)** |
|---|---|---|
| Hardware | Laptop you already own | Datacentre GPU |
| Deployment | On-premise, offline | Cloud only |
| Cost per conversation | **Zero** | Billed per GPU-minute, forever |
| Connectivity | Works without it | Constant uplink |
| Data | Never leaves the building | Sent to a third party |

**A cloud avatar bills for every minute of every conversation, forever. Alapon runs
on a machine you buy once.**

And the places that need this most — village schools, district offices, rural
clinics — have unreliable internet and no GPU budget. A 5 GB cloud model cannot be
deployed there at any price.

> **"A five-gigabyte model can't go to a village school. Alapon can run on the
> machine that's already sitting there."**

**Adding a new face:** ~4 minutes of video, ~7 hours of training, and the output is
a 49 MB file you can copy anywhere. That's what onboarding a customer's own
spokesperson costs.

---

## Where it goes

**Education — lead with this one.** More students than teachers, especially outside
the cities. An avatar that explains a concept in Bangla, at the student's pace,
without ever getting impatient. *Why a face:* it holds attention, and explanation
is conversational — you ask follow-ups.

**Government services.** Benefits, eligibility, forms — exactly where people give
up. *Why a face:* people will ask a face the same question three times. They will
not read the same paragraph three times.

**Banking and telecom.** Today it's IVR trees or English-first chat. Fastest path
to revenue, and the customer already has the hardware budget.

**Health information.** Not diagnosis — what a prescription means, when to visit a
clinic. Highest value, highest risk; needs clinical review and hard scope limits.

**Accessibility.** Voice in, voice out, with a visible speaking face. Lip movement
measurably aids comprehension — which is exactly why doing it *correctly* matters
rather than approximately.

The thread through all five: **a text box demands fluent reading and typing. A face
that speaks your language does not.**

> If they ask "which one?" — **pick education and go deep.** A list sounds like
> indecision.

---

## Why you should believe the numbers

**We quoted nobody.** We re-ran six published systems ourselves — Wav2Lip,
LatentSync, MuseTalk twice over, IP-LAP — on Bangla data *and* on HDTF, the
standard public English benchmark. Same code, same harness, every single cell.

- **78 measurements** across 6 speakers and 6 systems, with and without audio
- **Real video scored as a system** on both datasets — the row the field omits
- **Reproducible** — the scorer matches the original run to within 0.04 and gives
  the same answer on two different GPUs to within 0.004, so the results belong to
  the method and not the hardware
- **A Bangla corpus we built** — 2.78 hours, 24 speakers, 28 videos
- **A human study designed** — native Bangla speakers, blind, randomised — next to
  run

> **"We re-ran everything, including the measurement that makes our own numbers
> look less impressive."**

---

## Be honest about the limits

Say these before anyone asks. It costs nothing and buys everything.

- **Person-specific.** One face per model; ~4 minutes of video and ~7 hours to
  learn a new one. Not one-shot, and we don't claim it is.
- **We measure how far the mouth opens, not its shape.** Right amount, right time —
  yes. /a/ visibly distinct from /e/ — not yet. That's the next component.
- **The conversation intelligence is Gemini's.** Alapon is the real-time Bangla avatar
  layer, and every number above measures that layer.
- **Formal evaluation is on one Bangla speaker.** More speakers is the honest next
  step, and we know it.

---

## The ask

**A route to a pilot.** One school, one district office, or one call centre.

The system works and the numbers are measured. What it needs now is a real user in
a real setting — and the human study that turns our measurements into evidence
about what people actually perceive.

---

## Three minutes, if that's all you get

1. **240 million people. Zero products.** Six leading systems, none has seen
   Bangla. *(20s)*
2. **Demo.** Let it run. *(60s)*
3. **The broken ruler.** We scored real human video: 5.135. Wav2Lip scores 6.444 —
   25% better than a human, which is impossible. Alapon lands at 5.111, half a percent
   from real. *(40s)*
4. **The cost.** LatentSync needs 5 GB to reach where Alapon gets with 49 MB. It runs
   at 30 fps on this laptop using 0.3 GB of video memory. *(30s)*
5. **Why it matters.** That's the difference between needing a datacentre and
   running in a village school. *(20s)*

**Land on:** *"Same quality as a five-gigabyte model, at one percent of the size —
and it's the only one that speaks Bangla."*

---

## Objections

**"Isn't this just Gemini with a face?"**
Gemini writes the words. We built the layer that turns live audio into a
synchronised face at 30 fps on a 4 GB laptop GPU — and every number above measures
*that* layer against five published alternatives.

**"Your score is lower than Wav2Lip's."**
Wav2Lip's score is higher than a real human being's. That's the tell. The target is
proximity to real, not maximum score — and Wav2Lip was trained against this exact
scorer.

**"Then why not use the bigger model?"**
104× larger, cloud-only, and on the copying test it scores 0.359 against our 0.001.
It reaches its quality partly by copying the input video instead of listening.

**"How do you know it's actually good?"**
Three independent measurements: distance from real video, how much the mouth moves
versus a real person, and whether it moves when there's no audio at all. Human
study next.

**"Is it really the first?"**
The first we're aware of, and the first benchmarked against published systems on a
public dataset. None of the six leading systems we tested has been trained or
evaluated on Bangla.

**"How fast does it respond?"**
About 1.5 to 2 seconds from when you stop speaking. Most of that is deliberately
waiting to be sure you've finished — Bangla speakers pause mid-sentence, and cutting
people off is worse than a short wait. The avatar itself is not the bottleneck.

---

## Before you pitch — verify these

Don't quote a number you haven't re-checked.

- [x] ✅ **Live render speed measured on the RTX 3050** — 31–33 ms/frame, **30 fps**,
      **0.30 GB VRAM**, checkpoint confirmed **49.2 MB**. The live system's ~28 fps
      is consistent once audio and streaming are added. **Lead with these, not the
      2m33s batch figure.**
- [ ] ⚠️ **Never say "200 ms response time."** Your own README measures **~1.5–2 s**
      from *you stop talking* to *the avatar answers*. If 200 ms is real it measures
      something much narrower. Use the framing in the objections section instead —
      it's true and it's stronger.
- [ ] ⚠️ **Re-run the quality numbers after the clean retrain.** The lip-sync and
      mouth-movement figures come from a model that trained on part of its own test
      data. The fix is in; the retrain is ~15 hours. The **0.001 copying number is
      unaffected** — it comes from the architecture, not the training frames.
      *This is the one that could embarrass you. Do it.*
- [ ] ⚠️ **The "1080 costs ~10 ms/frame" comment didn't reproduce** — native 1080
      measured identical to 720. Don't cite the 720 choice as a saving. You may be
      able to ship 1080 for free.
- [ ] **Confirm the 240 million figure** and have a source ready.
- [ ] **Record a backup demo.** Live demos fail in front of judges.

*Measurement script: `scratchpad/bench_3050.py` — mirrors `inference_328`'s inner
loop with CUDA warm-up and synchronisation, 200 frames.*
