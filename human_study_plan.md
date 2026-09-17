# Human Evaluation Plan — Bangla Talking-Head Lip-Sync

**Two stages.** A 6-person pilot to validate the instrument, then a ≥15-person
main study to produce the result. They answer different questions and must not
be confused with each other.

---

## 0. What each stage is for

| | Pilot (n=6) | Main (n≥15) |
|---|---|---|
| Question | *Does this study work?* | *Which metric predicts perception?* |
| Output | go / fix / redesign | the correlation, the MOS table |
| Can it support a claim about systems? | **No** | Yes |

**A pilot with 6 raters cannot establish a correlation.** Do not report
"leakage correlates with MOS at n=6" — with ~20 stimuli and 6 raters the
confidence interval will span zero. Its job is to find out whether the task is
clear, the clips are the right length, raters agree with each other, and the
scale actually discriminates. If the instrument is broken, you want to know
before recruiting fifteen people.

---

## 1. The two questions being served

**FYDP question.** *Does the model produce correct Bangla lip shapes?*
Every metric measured so far is aperture — how far open the mouth is. Nothing
shows that /a/ looks different from /e/. Only a native speaker can say that,
and it is what the project proposal actually claims.

**Journal question.** *Which metric predicts what people perceive — LSE-C or
leakage?* Six controlled models differing 5x in leakage scored 4.88–5.14 on
LSE-C. Whether that blindness matters depends on whether humans notice the
difference.

One study serves both. The FYDP use is the obligation; the journal use is a
second analysis over the same ratings.

---

## 2. Stimuli

### Segment length: 8 seconds

Long enough to judge sync across several syllables, short enough that a rater
stays attentive. The 31 s clips already generated cut into 3 usable segments
each (skip the first second, which contains the reference-frame transition).

### Encode every clip identically

Same resolution, same CRF, same container, audio as AAC 128k. Otherwise part of
what raters judge is compression. Name them `clip_01` … `clip_NN` — **never**
by system. Filename leakage in a video player is the standard way blind studies
stop being blind.

### Pilot set — 20 clips, ~7 minutes

| source | clips | role |
|---|---:|---|
| Real redwan video | 2 | **upper anchor** |
| SyncTalk_2D as-released (rung 0) | 2 | **lower anchor** — leaks 0.42 |
| SyncTalk_2D `final_v2` | 3 | the system |
| Wav2Lip | 2 | high LSE-C, high leak |
| LatentSync 1.5 | 2 | diffusion |
| MuseTalk v1.5 | 2 | low leak, under-articulated |
| `final_v2` on **unseen** Bangla audio | 3 | the FYDP question |
| **silent-audio clips** (4 systems) | 4 | the leakage question |
| *repeats of 3 earlier clips* | *(+3)* | self-consistency check |

The anchors are the most important rows. **If real video does not score clearly
above the as-released model, the instrument is broken** and no other result from
the study means anything.

### Main set — scale to ~48 clips

Same construction, more segments per system, two source identities if a second
one is ready. Split across two sessions or two rater groups if it exceeds 15
minutes.

---

## 3. The task

Randomise clip order **per rater**. Allow replay. No system names anywhere.

For each normal clip, two questions, both 1–5:

**Q1. Lip-sync.** *Do the mouth movements match the speech?*
> 1 = completely mismatched · 3 = roughly right, noticeable errors · 5 = perfectly matched

**Q2. Naturalness.** *Does the face look real and natural?*
> 1 = obviously fake · 3 = acceptable · 5 = indistinguishable from real video

Keep these separate. A model can be perfectly synced and visually ugly;
conflating them makes both numbers vague.

For each **silent-audio** clip, one question instead:

**Q3.** *Is this person speaking?* — Yes / No / Unsure

That is the direct human read on leakage, and it is the most intuitive evidence
in the whole project: a person watching a silent clip and saying "yes, they are
talking" about a model that was fed nothing.

### Free text, once at the end

> *If any clip looked wrong, what specifically looked wrong?*

Native speakers will describe viseme errors in words no metric captures. This is
often the most useful data in the pilot.

---

## 4. Participants

**Native Bangla speakers.** Non-negotiable. The reason this defect was found at
all is that wrong Bangla visemes are visible to a native speaker and invisible
to everyone else. Non-native raters would wash out the effect being measured.

- **Pilot:** 6. Friends or labmates are fine at this stage.
- **Main:** ≥15, ideally none of whom saw the pilot.

Collect per rater: native language, prior exposure to AI-generated video,
device used, and whether audio was played through headphones or speakers.
Five minutes of demographics defends against "who were these people" in review.

---

## 5. Delivery

**Pilot — keep it crude.** A shared Drive folder of `clip_01.mp4` … plus a
Google Form with numbered questions. An hour to set up. Do not build software
for six people.

**Main — build the page.** A single page that plays one clip at a time,
randomises order per rater, records responses, and prevents skipping ahead.
Worth the effort at n≥15, not before.

---

## 6. Analysis

### Pilot — four checks, in order

1. **Anchor separation.** mean MOS(real video) − mean MOS(rung 0).
   **Need ≥1.0 on the 1–5 scale.** If the anchors do not separate, stop and
   redesign — nothing else is interpretable.
2. **Inter-rater agreement.** ICC(2,k) or Krippendorff's alpha across raters.
   **Need ≥0.5.** Below that the task is ambiguous, usually because the
   question wording is unclear or the clips are too short.
3. **Self-consistency.** Per-rater difference on the 3 repeated clips.
   A rater differing by ≥2 points on repeats is not attending; in the main
   study that is the exclusion criterion.
4. **Scale usage.** Histogram of all responses. If everything lands on 3–4,
   the scale is not discriminating and needs anchored examples shown up front.

**Also record:** median completion time, and any clip where several raters
commented on the same thing.

### Main — the result

```
Spearman( MOS_lipsync , LSE-C   )   across all clips
Spearman( MOS_lipsync , leakage )   across all clips
```

Score LSE-C and leakage **per 8-second segment**, not per system. Per-system
gives n≈6 and no statistical power. Per-segment gives n≈48 — and since LSE-C
was measured swinging from 2.167 to 6.484 across consecutive windows of one
video, that instability supplies the variation the correlation needs.

Three outcomes, all publishable:

- **Leakage correlates, LSE-C does not** → the proposed protocol is justified by
  evidence rather than by our own model winning.
- **Both correlate** → LSE-C is adequate; the critique softens to "incomplete".
- **Neither correlates** → the field has no validated metric at all. The largest
  finding of the three, and the most uncomfortable.

**Write down which you expect before running it.**

### FYDP analysis, separately

Mean MOS(lip-sync) for `final_v2` on unseen Bangla audio, against the real-video
anchor. Plus a thematic read of the free-text responses for viseme-specific
complaints — that is the evidence for "matches lip shape", which no number
currently supports.

---

## 7. Ethics

A written determination from the department is required before running it, even
for a 7-minute rating task with volunteers. It cannot be retrofitted after
review.

Include on the consent screen: purpose, duration, that responses are anonymous,
that participants may stop at any time, and a contact address. One paragraph.

---

## 8. Timeline

| | |
|---|---|
| Cut clips, encode, build form | 1 day |
| Ethics determination | start now — runs in parallel |
| Pilot (6 raters) | 2–3 days to collect |
| Pilot analysis + fixes | 1 day |
| Main study (≥15) | 1–2 weeks to collect |
| Analysis | 1 day |

The calendar, not the work, is the constraint. Recruiting is what takes weeks —
begin before the clips are ready.

---

## 9. What the pilot must not be used for

- Any claim about which system is better
- Any correlation between MOS and a metric
- Any number in the paper or the FYDP report

Its only outputs are: *the task works / the task needs fixing*, and a list of
fixes. Reporting pilot numbers as results is the most common way small studies
damage a paper.
