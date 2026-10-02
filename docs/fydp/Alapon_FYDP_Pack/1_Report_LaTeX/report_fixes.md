# Report fixes

Checked against `FYDP_Final_Report.pdf` (64 pages). Each fix gives the page, where on
the page, the exact line as it appears now, and what it should say instead.

**One rule for all fixes:** wherever a percent sign appears, the LaTeX source must
write it as `\%`. A bare `%` deletes the rest of that line from the PDF. That is what
broke the abstract.

Fixes are in page order. The ones marked **CRITICAL** state a result wrongly; fix
those first.

---

## Part A — Make the SyncNet fix count

The report describes the lip-sync expert fix but never shows that it improved anything.
These five fixes add that.

### A1 · Page 2 · Abstract · after the sentence about Alapon moving in 0.1%
**Add this sentence:**
> A second gap was the lip-sync expert that guides training: it had only ever seen
> matching audio and video, so it approved every mouth. Retrained on Bangla with
> mismatched audio, it now guides the model.

### A2 · Page 28 · Section 3.2, item 2 ("Bangla lip-sync expert") · end of the paragraph
**Now ends with:**
> …with the avatar being trained against this Bangla expert.

**Add after it:**
> The retrained expert can tell a correct mouth from a shifted one: its validation loss
> fell to 0.56, against 0.69 for guessing, where the original expert scored every audio
> offset near 1.0.

### A3 · Page 28 · Section 3.2, item 5 · last sentence
**Now:**
> Only the first one is measured directly (Section 4.3.1), the other three are design and
> training problems that we solved together.

**Replace with:**
> Two of them are measured: the mask (Table 4.2) and the lip-sync expert (Section 4.3.1).
> The other two were corrected together with them.

### A4 · Page 39 · Section 4.3.1 · paragraph under Figure 4.1 · CRITICAL
This is the only place the expert is linked to a result, and it is unreadable.

**Now:**
> The copying is only completely stopped when the jaw is fully covered, leaving even two
> rows visible. The LSE-C score is virtually unchanged and has no trend over all 6 models.
> These six models have a zero loss of lip-sync without the input of this lost
> information, whereas Alapon with it achieves 0.1%.

**Replace with:**
> Only covering the jaw completely stops the copying; leaving even two rows visible does
> not. Across all six models the LSE-C score barely changes and shows no trend.
>
> **The lip-sync expert.** These six models were trained without the lip-sync loss, and
> even the one with the jaw fully covered still moved its mouth in 8% of silent frames.
> Alapon adds the retrained Bangla lip-sync expert and reaches 0.1%. The original expert
> could not have done this: it approved every mouth, so it gave the model nothing to learn
> from. Alapon also trained for longer, so the drop is not due to the expert alone.

### A5 · Page 46 · Discussion list · insert as a new item 2
**Add:**
> 2. **The lip-sync expert now teaches.** The original expert approved every mouth.
> Retrained on Bangla with mismatched audio, it separates in-sync from out-of-sync speech,
> and with it the mouth movement during silence fell from 8% to 0.1%.

Then renumber the items after it (3–6).

---

## Part B — Errors introduced by the rewording

The easiest fix for B1–B5 is to copy the original paragraph back from
`latex_source/` (in this folder), which has none of these errors.

### B1 · Page 2 · Abstract · first paragraph · garbled
**Now:**
> Commercial Cloud services now boast Bangla voices, which are closed, exist only in the
> cloud, and are published. None of the open research systems were able to evaluate their
> lips in relation to Bangla speech. the one that had been trained /evaluated in Bangla was
> examined. This project creates a real-time, open source, system. She has a
> conversational avatar which can talk in Bangla and runs her avatar on a laptop. The user
> speaks A speech-to-speech agent that replies in Bangla using Google Gemini is available
> in the browser .Google Gemini speech-to-speech agent that replies in Bangla is available
> in the Web Browser .

**Replace with:**
> Commercial cloud services now offer Bangla voices, but they are closed, run only in the
> cloud, and publish no evaluation of how well their lips follow Bangla speech; none of
> the open research systems we examined had been trained or evaluated on Bangla. This
> project builds an open, real-time conversational avatar that speaks Bangla and runs on a
> laptop. The user speaks Bangla in a web browser, and a speech-to-speech agent built on
> Google Gemini replies in Bangla.

### B2 · Page 2 · Abstract · second paragraph · CRITICAL
Two `%` signs deleted text, and one sentence reverses the result.

**Now:**
> …and seconded it by performing published lip-sync systems using the same Bangla clip.
> …The worst was that it would show its jaw through the mouth mask, as the model would
> copy mouth. It moved its mouth 41Played for the time the audio was not playing. We have
> a new model, Alapon, that moves in 0.1This is because when it hears a Bangla speaker it
> has to open its mouth 2.3 times more than when it hears a // speaker . This is 1.7 times
> worse than it was prior to the change. It was trained on overlapping training data and
> test-clip, thus its test-clip After a complete retraining, scores are being re-measured.

**Replace with:**
> …and confirmed the choice by testing published lip-sync systems on the same Bangla clip.
> …The most serious was that its mouth mask left the jaw visible, so the model copied mouth
> movement from the input video instead of following the audio: its mouth moved in 41% of
> frames when the audio was silent. Our improved model, Alapon, moves in 0.1%. Driven by
> six Bangla speakers it had never heard, its mouth opens 2.3 times wider on আ than on
> প/ব/ম, against 1.7 times before the change. Its training data overlapped the test clip,
> so its test-clip scores are being re-measured after a clean retrain.

### B3 · Page 2 · Abstract · third paragraph
**Now:**
> Five of the six leaked at all on one of the sets, they moved their mouths It performed
> well in up to 46% of the silent frames…

**Replace with:**
> Five of the six leaked on at least one dataset, moving their mouths in up to 46% of
> silent frames…

### B4 · Page 10 · Section 1.4 · Stage 5
**Now:**
> This was attributed to the jaw being visible through the model's mouth mask, which
> allowed for the night vision of the eyes. In the case of the model, she copied the
> movements of her mouth, rather than listening to the sound. We found three additional
> gaps in the model design and training were highlighted and were solved by all of four.
> The result is Alapon, which we tested lip-sync, silence, reconstruction quality and
> individual Bangla. sounds.

**Replace with:**
> We traced this to the model's mouth mask, which left the jaw visible, so the model
> copied mouth movement from the video instead of listening to the audio. We found three
> further gaps in the model's design and training and addressed all four. The result is
> Alapon, which we evaluated on lip-sync, silence, reconstruction quality and individual
> Bangla sounds.

### B5 · Page 10 · Section 1.4 · Stage 6
**Now:**
> The same silence test was applied to six other schools. Existing published systems on
> Bangla clip and public English dataset HDTF. Five of the It was also leaked, as were six,
> and the de facto lip-syncing metric didn't pick it up.

**Replace with:**
> We applied the same silence test to six published systems on the Bangla clip and on the
> public English dataset HDTF. Five of the six leaked too, and the standard lip-sync metric
> did not detect it.

### B6 · Page 10 · Section 1.4 · Stage 4
**Now:**
> We have used Bangla video for training SyncTalk 2D and retrained it on the same video.
> It has its lip-sync expert which was developed in Bangla…

**Replace with:**
> We trained SyncTalk_2D on Bangla video, retrained its lip-sync expert on Bangla…

### B7 · Page 28 · Section 3.2, item 7 · second sentence
**Now:**
> A visually distinct mouth shape may represent more than one phoneme, for example,
> viseme 'a'a may represent both 'a' and 'u'.

**Replace with:**
> Several phonemes can share one viseme; for example, প, ফ, ব, ভ and ম all close the lips.

('a' and 'u' are different visemes in Table 3.3, so the current example contradicts the table.)

### B8 · Page 45 · Section 4.3.4, finding 3 · CRITICAL
**Now:**
> In Table 4.2, covering the jaw reduced the leakage by 39% (to 8%), but LSE-C did not change.

**Replace with:**
> In Table 4.2, covering the jaw cut leakage from 39% to 8%, but LSE-C did not change.

### B9 · Page 45 · Section 4.3.4, finding 5
**Now:**
> MuseTalk 1.5's mouth did not move in silence for 99.6% of the frames in the Bangla clip
> and 54% of the frames in the HDTF clip.

**Replace with:**
> MuseTalk 1.5 moved its mouth during silence in 0.4% of frames on the Bangla clip but 46%
> on HDTF.

### B10 · Page 45 · Section 4.3.5 · first line under Table 4.7 · CRITICAL
**Now:**
> The avatar's framerate is 25 frames per second and consumes less than 10% of the
> laptop's GPU memory.

**Replace with:**
> The avatar renders at 30 frames per second, faster than the 25 it needs, using less
> than 10% of the laptop's GPU memory.

### B11 · Page 46 · Section 4.3.5 · first line of the page
**Now:**
> Gemini responds quickly throughout an entire conversation.Gemini responds effortlessly
> throughout a long conversation.

**Replace with:**
> Gemini's reply time stays flat across a long conversation.

### B12 · Page 46 · Discussion, item 3
**Now:**
> Alapon is small enough and fast enough to render at 30 fps on a laptop, as is a 5 GB
> model of real-video, which delivers the same score in video-lip-sync.

**Replace with:**
> At 49 MB, Alapon reaches the real-video lip-sync score, as a 5 GB model does, and
> renders at 30 fps on a laptop.

### B13 · Pages 46–47 · Section 4.4 Summary · CRITICAL
This reverses the main result.

**Now:**
> The contrast between Bangla open and closed sounds was increased from 3.0 to 3.8 with
> the cover of the jaw while keeping the movement of mouth during silence at 41% for the
> dummy speaker, which was maintained at 4.0 for the real speaker.

**Replace with:**
> Covering the jaw reduced mouth movement during silence from 41% to 0.1% and raised the
> contrast between open and closed Bangla sounds from 3.0× to 3.8×, against 4.0× for the
> real speaker.

**Also in the same summary, now:**
> …and the system responds 1.5 2 seconds.

**Replace with:** "…and the system answers within 1.5–2 seconds."

### B14 · Page 57 · Section 6.1 · first paragraph · CRITICAL
**Now:**
> A user types Bangla into a web browser and the agent responds in a formal style of
> Bangla, with a 2D avatar saying the reply, which is synchronised with the user's typing.

**Replace with:**
> A user speaks Bangla in a web browser, a speech-to-speech agent replies in formal
> Bangla, and a 2D avatar speaks the reply with synchronised lip movement.

### B15 · Page 57 · Section 6.1 · second paragraph
**Now:**
> The most critical was the mouth mask: The mask that was copied was visible jaw, not the
> listening one, and its mouth was moving in 41 percent of the frames in the silence.

**Replace with:**
> The most important was the mouth mask: the model copied the visible jaw instead of
> listening, and its mouth moved in 41% of frames during silence.

### B16 · Page 7 · List of Tables, Table 4.4 (and its caption on page 34)
**Now:** "Open  closed with six new Bangla speakers" (the ÷ sign is missing)
**Replace with:** "Open ÷ closed with six new Bangla speakers"

### B17 · Page 6 · List of Figures, Figure 3.2
**Now:** "Sequence diagram of the proposeds system"
**Replace with:** "Sequence diagram of the proposed system"

### B18 · Page 7 · List of Tables, Table 3.1
**Now:** "Created candidate avatar models that meet project requirements"
**Replace with:** "Candidate avatar models against the project requirements"

---

## After fixing

- Search the PDF for "%" and check that each number is followed by the rest of its sentence.
- Check that ×, ÷ and → show up in Tables 4.3–4.4 and the Discussion; a missing glyph can look like a blank.
- If you use a rewording tool again, re-check every number against the original afterwards.
