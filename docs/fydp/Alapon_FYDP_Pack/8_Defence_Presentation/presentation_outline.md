# Alapon — FYDP defence presentation outline

**Length:** 10–12 minutes · **17 main slides + 5 backup slides**
**Team:** Group E1 · Dept. of CSE, United International University
**Supervisor:** Dr. Mohammad Nurul Huda · **Co-supervisor:** Azizur Rahman Anik

---

## Design brief (paste this into Claude Design first)

> A 16:9 academic defence deck for **Alapon**, a real-time Bangla talking avatar
> that runs on a laptop. Clean and light: white background, plenty of space, one
> message per slide, very little text. The **slide title is the takeaway sentence**,
> not a topic label ("The mouth should stay still when nobody speaks", not "Results").
>
> **Colours** — use these meanings consistently on every slide:
> - Blue `#2A78D6` = Alapon / our work / good results
> - Orange `#EB6834` = the problem / the leak / the old model
> - Warm grey `#C9C7BF`, text grey `#52514E` = everything else
> - Near-black text `#121A2A`
>
> **Fonts:** Anek Bangla for titles (it has Latin and Bangla, so "Alapon" and
> "আলাপন" match), Atkinson Hyperlegible for body text, IBM Plex Mono for small labels.
>
> A thin progress bar at the top shows the 7 sections: Introduction · Related work
> · Scope · Contributions · Methodology · Outcome · Limitations.
> Slide 2 is a formal **Problem statement** slide: one large sentence, then three
> sub-problem tiles; slide 7 (Contributions) answers them in the same order.
> Big numbers are the visual anchor where the slide is about a result.
> Use the provided figures as they are; don't redraw charts.

**Figures:** all in `Alapon_FYDP_Pack/`. Use the *poster* versions
(`4_Figures_for_Poster/`, `5_Diagrams/`) on slides; they carry large type.
Where a figure has its own title, crop it off or let the slide title replace it.

---

## The story in one line

> No open Bangla talking avatar existed → we built one that runs on a laptop →
> while building it we found the model was copying the jaw instead of listening →
> we fixed it (**Alapon**) and showed it follows Bangla sounds → the same leak
> exists in published systems, and the standard metric can't see it → journal paper in progress.

State the problem on slide 2, answer it point by point on slide 7, and close with
this line on slide 17. That is the big picture.

---

## Timing plan

| Section | Slides | Time |
|---|---|---|
| Title | 1 | 0:20 |
| 1. Introduction (problem statement, what we built) | 2–3 | 1:30 |
| 2. Related work | 4–5 | 1:15 |
| 3. Scope | 6 | 0:40 |
| 4. Contributions | 7 | 0:50 |
| 5. Methodology | 8–12 | 3:35 |
| 6. Outcome | 13–16 | 3:15 |
| 7. Limitations & future work | 17 | 0:50 |
| Thank you / Q&A | 18 | — |
| **Total** | | **≈ 12:15** — rehearse to 11:30 by keeping slides 4–5 to one minute together |

If you run long, shorten slides 5 and 12 first. Never cut the demo (slide 16).

---

## Slide 1 — Title  ·  0:20

**On slide**
- **Alapon** আলাপন
- Real-Time Bangla Speech-Driven Avatar Generation
- Group E1 — six member names with IDs
- Supervisor, co-supervisor, course teacher · UIU CSE · FYDP defence, [date]

**Visual:** a single still frame of the avatar speaking, right side.

**Say:** "We built Alapon, a talking avatar that listens and answers in Bangla, in real time, on a laptop."

---

# 1 · INTRODUCTION

## Slide 2 — Problem statement  ·  0:50

**Title:** Problem statement

**The statement (large, centred, one sentence):**
> Bangla speakers have no open, real-time conversational avatar that runs on
> affordable hardware and whose lips are shown to follow Bangla speech.

**Three sub-problems (three tiles below, orange icons):**
1. **Language:** talking-head models are trained and tested almost only on English. None of the open research systems has been trained or evaluated on Bangla.
2. **Hardware:** the best-looking models need datacentre GPUs; schools, offices and clinics have a laptop at most.
3. **Trust:** cloud products offer Bangla voices, but they are closed, paid per minute, and never show that the lips actually follow the voice.

**Footer line (blue):** Our goal: an open Bangla avatar that runs on a laptop, with lips we can *prove* follow the audio.

**Visual:** three tiles in a row with simple icons (a speech bubble with "EN", a server rack, a question mark over lips). Orange for the problems, the blue footer as the answer.

**Say:** read the one-sentence statement, then one line per tile. End with: "Each of these three problems maps to one of our contributions" (slide 7 answers them in the same order).

## Slide 3 — What we built  ·  0:40

**Title:** You speak Bangla. A face answers in Bangla, in under two seconds.

**On slide:** four steps in a row with icons
1. User speaks Bangla in the browser
2. Gemini understands and replies in Bangla speech
3. **Alapon** turns the reply into lip-synced video on the laptop GPU
4. The browser plays the face and voice together

**Key numbers (strip at the bottom):** 49 MB model · 30 fps on an RTX 3050 laptop · 1.5–2 s response

**Visual:** `5_Diagrams/fig3_1_system_diagram.png` (simple version).

---

# 2 · RELATED WORK

## Slide 4 — Two families of talking heads  ·  0:40

**Title:** 3D models look best; 2D models are the only ones fast enough for a laptop

**On slide:** two columns
- **3D (NeRF, Gaussian Splatting):** AD-NeRF, SyncTalk, GaussianTalker, TalkingGaussian, InsTaG. High realism, but heavy GPUs and long training
- **2D lip-sync:** Wav2Lip, IP-LAP, MuseTalk, LatentSync, SyncTalk_2D. Edit only the mouth of real video; smaller and faster
- One line under both: *none trained or tested on Bangla*

**Visual:** a two-column comparison; SyncTalk_2D highlighted in blue as "what we build on".

## Slide 5 — The gap, as a table  ·  0:35

**Title:** No existing system covers Bangla, a laptop, and a check that the lips follow the audio

**On slide:** a simplified version of report Table 2.1. Rows: *Bangla training · Bangla lip-sync expert · Runs on a laptop GPU · Silent-audio leak test*. Columns: SyncTalk · GaussianTalker · InsTaG · TalkingGaussian · NeRFFaceSpeech · **Alapon**. Grey ✗ everywhere except a blue ✓ column for Alapon.

**Say:** "KeySync (2025) proposed testing with silent audio. We use that test, and it becomes central to our story."

---

# 3 · SCOPE

## Slide 6 — What this project does and does not do  ·  0:40

**Title:** Scope: one Bangla speaker's face, any Bangla conversation, on a laptop

**On slide:** two columns
- **In scope:** real-time Bangla conversation · a person-specific 2D avatar (one speaker, ~5 min of video) · runs on a 4 GB laptop GPU · measured against published systems
- **Out of scope:** any-face (one-shot) avatars · 3D head motion and expressions · offline use without the cloud conversation model · a human perception study (planned, see slide 17)

---

# 4 · CONTRIBUTIONS

## Slide 7 — Four contributions  ·  0:50

**Title:** What we contribute

**On slide:** four cards, blue numbers. A small orange tag on each card links it back to the problem statement (slide 2).
1. **A working real-time Bangla avatar system** that runs on a laptop · *answers: Hardware*
2. **Alapon:** SyncTalk_2D adapted to Bangla, with **four gaps in the model found and closed** · *answers: Language*
3. **A Bangla sound-level evaluation:** does the mouth close on প ব ম and open on আ? · *answers: Trust*
4. **A finding about the field:** 5 of 6 published systems leak, and the standard lip-sync score cannot see it (journal paper in progress) · *answers: Trust*

Small footer: + a 2.78-hour, 24-speaker Bangla audiovisual corpus

---

# 5 · METHODOLOGY

## Slide 8 — System architecture  ·  0:40

**Title:** One conversation turn, end to end

**On slide:** the numbered flow (1 token → 2 join → 3 user audio → 4 end of turn after 0.55 s silence → 5–6 Gemini → 7 reply audio to Alapon → 8 frames back → 9 to the browser)

**Visual:** `5_Diagrams/system_architecture.png` (detailed poster version, steps numbered 1–9).

**Say:** three Bangla-specific fixes in the agent: transcription pinned to `bn-BD` (it was switching to Hindi or Punjabi script); a local voice detector waits 0.55 s so Bangla pauses aren't cut off; technical words given as hints.

## Slide 9 — Choosing the model  ·  0:35

**Title:** Only one candidate fits a 4 GB laptop GPU

**On slide:** the size chart; one line: "We chose SyncTalk_2D and confirmed it by testing every alternative on the same Bangla clip."

**Visual:** `4_Figures_for_Poster/fig_model_size.png`

**Say:** trade-off: it must be trained per face (~5 min of video), which suits an avatar of one fixed speaker.

## Slide 10 — Training data  ·  0:40

**Title:** Trained on 5 minutes of one Bangla speaker

**Stats strip (top):** 7,717 frames · 25 fps · 1080 × 1080 · 16 kHz audio · recording `redwan` (team member Sheikh Redwanul Islam)

**Visual:** `8_Defence_Presentation/fig_redwan_data.png`, the whole slide body. It shows the seven processing steps:
1. **Source frame:** the video, converted to 25 fps
2. **Face landmarks:** 110 points found on every frame
3. **Face crop:** a square cut from the landmarks, 320 × 320
4. **Model input:** the mouth *and jaw* blacked out; the model must repaint them
5. **Audio track:** 16 kHz, taken from the same video
6. **Mel spectrogram → audio features:** one feature vector per video frame
7. **Split:** training 0–6170 (4.1 min) · validation 31 s · test 31 s, in continuous blocks

**Say:** "Only five minutes of video, which is why it runs on a laptop. The split is in
continuous blocks because neighbouring frames are almost identical; a random split would
leak test frames into training." Point at step 4: "This is where the jaw fix lives." That
sets up slide 12.

## Slide 11 — Building Alapon  ·  0:50

**Title:** Alapon: SyncTalk_2D adapted to Bangla, and its lip-sync teacher fixed

**Left: the lip-sync expert gap (the highlight of this slide)**
- **Gap:** the original lip-sync expert only ever saw *matching* audio and video, so it
  approved every mouth. Its score stayed near 1.0 at every audio offset, so it taught the model nothing.
- **Fix:** retrained on Bangla with *mismatched* audio, so it learns what "out of sync" looks like
- **Result:** our jaw-covered models trained without it still moved in **8%** of silent
  frames; Alapon, with the fixed expert, moves in **0.1%**

**Right: two more changes, small text**
- Streaming audio features, identical to offline, so the lips don't drift in live use
- Tried a multilingual encoder (XLS-R): **didn't work**, so not used

**Visual:** `5_Diagrams/alapon_model_architecture.png` (the "Bangla lip-sync expert" box
in its training panel is the one this slide is about). Add a mini before/after: orange
"expert approves everything → 8%", blue "fixed expert → 0.1%".

**Say:** "A teacher that says 'good' to every answer can't teach. The original expert was
exactly that. Once it could tell right from wrong, the silence leak dropped from 8% to
0.1%." Keep the wording as *with* the fixed expert, not *because of* it alone: those
models also differ in training length and settings.

## Slide 12 — The gap we found in the model  ·  0:50  ← *the turning point*

**Title:** The avatar was copying the jaw instead of listening

**On slide**
- The model repaints a mouth that is blacked out, but the mask **left the jaw visible**
- It learned to read the jaw: that 10-pixel strip caused **91% of mouth motion**
- Result: the mouth moved even when the audio was silent
- **Fix:** hide the jaw too, plus three training fixes

**Visual:** `5_Diagrams/fig3_6_alapon.png` (the mask inset: orange "jaw visible" vs blue "jaw hidden"), next to `4_Figures_for_Poster/fig_jaw_rows.png`

**Say:** "We trained six models that differ only in how much jaw they see. Only hiding all of it stops the copying, and the standard lip-sync score doesn't change at all."

---

# 6 · OUTCOME

## Slide 13 — Silence  ·  0:45

**Title:** The mouth now stays still when nobody is speaking

**Big number:** **41% → 0.1%** of silent frames with the mouth moving

**Visual:** `4_Figures_for_Poster/fig_silence_bangla.png`

**Say:** Wav2Lip GAN 46% and LatentSync 36% leak too. MuseTalk is also still, but it barely moves during speech either; Alapon moves 94% as much as the real speaker.

## Slide 14 — Bangla sounds  ·  0:50

**Title:** Lips close on প ব ম and open on আ, like a real speaker

**Big numbers:** **3.8×** (Alapon) vs **4.0×** (real speaker) · **1.7× → 2.3×** on six voices it never heard

**Visual:** `4_Figures_for_Poster/fig_bangla_sounds.png` (left) + `fig_six_speakers.png` (right)

**Say:** "A Bangla speech recogniser times every letter; we measure how far the mouth opens on each. On six new Bangla speakers it improved on every one."

## Slide 15 — What we found about the field  ·  0:50

**Title:** The standard lip-sync score rates some fakes above real people

**On slide**
- A real speaker is perfectly in sync with their own voice, so no system should beat real video
- On HDTF, **4 of 6** systems score above real video; for Wav2Lip it is statistically significant
- **5 of 6** published systems leak on at least one dataset, and the score doesn't notice
- → **Journal paper in progress:** *Evaluation Integrity in Audio-Driven Talking-Head Generation* (target: IEEE Transactions on Multimedia)

**Visual:** `4_Figures_for_Poster/fig_lsec_vs_real.png`

## Slide 16 — Live demo  ·  0:50

**Title:** Alapon, live

**On slide:** the demo itself (or the recorded backup video). A small corner strip: 30 fps · 0.3 GB of 4 GB · 49 MB · 1.5–2 s.

**Say:** ask one short question in Bangla. Let it answer; don't talk over it.
**Always have the recorded demo video ready** in case the live one fails.

---

# 7 · LIMITATIONS & FUTURE WORK

## Slide 17 — Limitations and what's next  ·  0:50

**Title:** What it doesn't do yet, and what's next

**Limitations (left)**
- One speaker's face; a new face needs new training
- Mouth *opening* matches Bangla; mouth *shape* (rounded vs spread) not yet shown
- Weaker on new voices (2.3×) than on its own speaker (4.0×)
- Needs the cloud for the conversation model
- Test-clip scores come from a model whose training overlapped the test clip; a clean retrain is running (the silence and six-voice results are not affected)
- No human study yet

**Next (right)**
- Journal paper · human study with ≥15 native Bangla speakers
- Train lip *shape* from our Bangla phoneme-to-viseme map
- Multi-speaker training on the corpus · pilot in a Bangla-medium school

**Closing line:** repeat the one-line story.

## Slide 18 — Thank you

**Alapon** আলাপন · Group E1 · "Questions?" · a QR code linking to the demo video.

---

## Backup slides (only if asked)

**B1 — "Did it train on the test data?"**
Yes. We found it ourselves and disclosed it. The silence test and the six-new-voice test are unaffected; a clean retrain is in progress.

**B2 — "HeyGen already does Bangla."**
It's closed, cloud-only and paid per minute, and publishes no Bangla lip evaluation. Ours is open, runs on a laptop, and is measured on Bangla.

**B3 — "Why is your lip-sync score not the highest?"**
Because the highest scores are *above real video*, which is impossible for a truly synced mouth. Wav2Lip was trained against the same scorer. Alapon matches real video (5.11 vs 5.14).

**B4 — Full comparison table**
Report Table 4.5 (all systems: size, LSE-D, LSE-C, silence, articulation) and `fig_leak_two_datasets.png`.

**B5 — The corpus**
`fig_corpus_speakers.png`: 24 speakers, 2.78 h. *Show only if asked; the speakers did not individually consent.*

---

## Rehearsal checklist
- [ ] Run through with a timer: aim for **11 minutes** so questions don't cut you off
- [ ] Each member knows which slides they present
- [ ] Recorded demo video on the laptop and on a USB stick
- [ ] Test the live demo on the venue's internet 30 minutes before
- [ ] Everyone can explain slide 12 (the jaw) in one sentence: that is the question judges will ask
