# Real-Time Bangla Speech-Driven Avatar Generation

**By Group E1**

| Name | ID |
|---|---|
| Sahil Al Farib | 011222186 |
| Momota Ahsana Meem | 011222254 |
| Sheikh Redwanul Islam | 011222142 |
| Khan Md. Shams Arefin | 011222320 |
| Sumaiya Akter Eva | 011222188 |
| Rumman Hossain | 011222308 |

**Course Teacher:** Dr. Riasat Azim, United International University, Dhaka
**Supervisor:** Dr. Mohammad Nurul Huda, United International University, Dhaka
**Co-Supervisor:** Azizur Rahman Anik, United International University, Dhaka

*Submitted in partial fulfilment of the requirements of the degree of Bachelor of
Science in Computer Science and Engineering*

Department of Computer Science and Engineering
United International University
September 26, 2026

---

> **Editor's note — delete this box before submission.**
> Five items need information only the team has. Search for "TODO"; none may
> remain in the submitted version:
> (1) a Bangla phonetics check of Table 3.3; (2) consent for the `redwan`
> recording (Section 5.1.4); (3) the Gemini API cost and (4) the Colab spend
> (Table 5.1); (5) the figures, marked **[Figure x.y: … — insert image]**. Reuse
> the FYDP-1 images except Figures 3.1, 3.2 and 3.5, which must be redrawn to
> match Section 3.2.

---

## Abstract

Real-time talking-head avatars exist almost only for English; no published system
we examined had been trained or evaluated on Bangla. This project closes that gap
with a real-time conversational avatar that speaks Bangla. The user speaks Bangla
in a web browser, a speech-to-speech agent built on Google Gemini replies in
Bangla, and a 2D avatar speaks the reply with synchronised lip movement.

To choose the avatar model we tested published lip-sync systems on Bangla video
and selected SyncTalk_2D, the only one small and fast enough for a laptop. We
adapted it to Bangla with Bangla training video, a Bangla-trained lip-sync expert
and streaming audio features, and tuned speech recognition and turn detection for
Bangla. While doing so we found four gaps in the SyncTalk_2D model that limited
its performance. The most serious was that its mouth mask left the jaw visible, so
the model copied mouth movement from the input video instead of following the
audio: its mouth moved in 41% of frames when the audio was silent. Our improved
model, **Alapon**, moves in 0.1%,
reconstructs the face more accurately (PSNR 29.8 → 33.1 dB), and separates open
and closed Bangla sounds almost as well as the real speaker (3.8× against 4.0×).

We then applied the same silence test to six published systems on Bangla video
and on the public English dataset HDTF. Five of the six leaked on at least one
dataset, moving their mouths in up to 46% of silent frames, and the standard
lip-sync metric could not detect it — it even scored three systems significantly
*above* real video. These findings are being written up as a journal paper.

The completed system runs Alapon, a 49 MB model that renders at 30 frames per
second on a laptop RTX 3050 using 0.3 GB of GPU memory, and answers within
1.5–2 seconds. Real-time Bangla avatar conversation is therefore feasible on
consumer hardware, with applications in education, public services and
accessibility.

---

## Acknowledgements

We begin by expressing our deepest gratitude to Almighty Allah, by whose grace we
have come this far, and to the many people whose encouragement made this work
possible.

We are profoundly grateful to our supervisor, Dr. Mohammad Nurul Huda, who guided
us and gave constructive feedback throughout this work, and who encouraged us to
report our findings on mouth-movement leakage and to develop them into a journal
paper. We thank our course instructor, Dr. Riasat Azim, for his academic
guidance, useful recommendations and inspiration throughout the trimester. We owe
a great deal to our co-supervisor, Mr. Azizur Rahman Anik, for his time, patience
and practical advice, which considerably improved the quality of our work. We are
also grateful to our peers and friends for their discussion, cooperation and
support.

Finally, and most importantly, we thank our families, particularly our parents,
for their unconditional love and constant support. Their faith in us is what kept
us going.

---

## Table of Contents

1. [Introduction](#chapter-1-introduction)
2. [Background](#chapter-2-background)
3. [Project Design](#chapter-3-project-design)
4. [Implementation and Results](#chapter-4-implementation-and-results)
5. [Standards and Design Constraints](#chapter-5-standards-and-design-constraints)
6. [Conclusion](#chapter-6-conclusion)
7. [References](#references)

---

# Chapter 1: Introduction

This chapter introduces the problem of real-time conversational avatars for
Bangla, the motivation for the project, its objectives and methodology, and the
outcome that was delivered.

## 1.1 Project Overview

Advances in speech recognition, large language models and speech synthesis have
made voice assistants such as Alexa, Siri and Google Assistant part of everyday
life. In parallel, audio-driven talking-head models can now generate a
photorealistic face whose lips follow an arbitrary speech signal. Combining the
two gives a *conversational avatar*: an assistant that answers with a face as well
as a voice. That is more engaging, and more accessible to people who rely on
visual speech cues, than audio alone.

Almost all of this technology is built for English. The talking-head systems most
widely used in research are trained on English video and evaluated on English
benchmarks, and none of the published systems we examined had been trained or
evaluated on Bangla. Bangla speakers therefore have no real-time conversational
avatar that speaks their language.

This project builds one. The user speaks Bangla into a web browser; a
conversational agent understands the request and replies in spoken Bangla; and a
2D avatar renders that reply with synchronised lip movement, all in real time.
The avatar model, **Alapon**, is SyncTalk_2D [18] adapted to Bangla and improved,
and the whole avatar pipeline runs on a consumer laptop GPU.

Building Alapon also produced a research result. The gap we closed in SyncTalk_2D
— a mouth that copies the input video instead of following the audio — turned out to be
present in five of the six published lip-sync systems we tested, and invisible to
the standard lip-sync metric. Those findings are being prepared as a
journal paper (Section 6.3.1).

## 1.2 Motivation

Existing AI communication systems handle Bangla poorly. They are configured
around English, so Bangla speech is often transcribed into the wrong script or
misunderstood, and the few avatar systems that exist do not generate Bangla lip
movement at all. Audio-only assistants give no visual speech cues, which reduces
engagement and excludes users who rely on lip reading. Most avatar systems also
depend on large models running on datacentre GPUs, which makes them expensive and
impossible to deploy where connectivity is limited.

A real-time Bangla avatar addresses all of these at once. A face that speaks the
user's language is easier to engage with than a text interface, particularly for
users who are not comfortable reading or typing. A model small enough to run on a
consumer laptop can be deployed in schools, public offices and clinics that have
neither reliable internet nor a GPU budget. The same system supports education,
public-service guidance, customer support and assistive technology.

For such a system, the lips must genuinely follow the speech. A mouth that moves
when nobody is speaking, or that copies movement from the source video, looks
wrong to a native speaker and defeats the purpose of a visible face. Checking this
directly therefore became part of the project.

## 1.3 Objectives

The aim of the project is to build a real-time conversational avatar that speaks
Bangla with accurate, audio-driven lip synchronisation. The specific objectives
are:

1. To build an end-to-end, real-time voice-to-voice-and-video conversational
   pipeline that understands and answers in Bangla.
2. To select an avatar model by testing published systems against the project's
   requirements, adapt it to Bangla speech, and verify that its mouth motion is
   genuinely driven by the audio.
3. To collect and prepare Bangla audiovisual data for training and evaluation.
4. To keep the avatar within a real-time budget of 25 frames per second on
   consumer hardware.
5. To evaluate the result objectively against published systems, and to report
   what that evaluation reveals about the systems and their metrics.

## 1.4 Methodology

The project followed the sequence below. Each stage fed the next.

**Stage 1 — Identifying the gap.** We reviewed audio-driven talking-head
generation and similar commercial applications (Chapter 2). None had been trained
or evaluated on Bangla, and most needed GPUs far beyond a laptop.

**Stage 2 — Building the real-time system.** We first built the complete pipeline
with an English-speaking agent. This separated the engineering problems of
latency, streaming and audio–video synchronisation from the linguistic problems
of Bangla, and confirmed that the architecture could hold the real-time budget.
We then built the Bangla agent on Google Gemini in speech-to-speech mode, with
transcription pinned to Bangla, turn detection retuned for Bangla speech rhythm,
and a formal-register Bangla persona.

**Stage 3 — Testing and selecting the avatar model.** We tested published
lip-sync systems on the same Bangla clip and compared them on size, speed and
lip-sync. SyncTalk_2D was the only candidate small and fast enough to run in real
time on a 4 GB laptop GPU, so we selected it (Section 3.2).

**Stage 4 — Adapting and improving the model.** We trained SyncTalk_2D on Bangla
video, retrained its lip-sync expert on Bangla, and built a streaming version of
its audio features for live use. We also tried a multilingual speech encoder
(XLS-R), which did not work.

**Stage 5 — Finding and fixing the leak.** The trained avatar moved its mouth
when the audio was silent. We traced this to the model's mouth mask, which left
the jaw visible, so the model copied mouth movement from the video instead of
listening to the audio. We found three further gaps in the model's design and
training and addressed all four. The result is **Alapon**, which we evaluated on lip-sync, silence,
reconstruction quality and individual Bangla sounds.

**Stage 6 — Checking the leak in other models.** We applied the same silence
test to six published systems on the Bangla clip and on the public English
dataset HDTF. Five of the six leaked too, and the standard lip-sync metric did
not detect it. This became the basis of a journal paper, now in progress.

**Stage 7 — Completing the system.** Alapon was integrated into the live system
and its speed and response time were measured on the target laptop.

> **[Figure 1.1: Graphical Abstract — insert image]**

## 1.5 Project Outcome

The project delivers a working real-time Bangla conversational avatar. Its main
outcomes are:

- A complete, running system: browser interface, real-time media layer, Bangla
  conversational agent and avatar rendering server.
- **Alapon**, a Bangla avatar model built on SyncTalk_2D, 49 MB in size,
  rendering at 30 frames per second on a laptop RTX 3050 with 0.3 GB of GPU
  memory.
- Four gaps found and closed in the SyncTalk_2D model. The mouth-mask fix
  stops the mouth moving during silence (41% of frames → 0.1%) and improves
  reconstruction (PSNR 29.8 → 33.1 dB).
- A Bangla phoneme-to-viseme mapping and a Bangla phoneme-level evaluation:
  Alapon's lips close on প/ব/ম and open on আ closer to the real speaker than any
  other system tested, including for six Bangla speakers it had never heard.
- A benchmark of six published systems on Bangla and English video with one
  scoring code, showing that five of the six leak on at least one dataset and
  that the standard lip-sync metric scores three of them above real video.
- A journal paper in preparation on these findings.
- A Bangla audiovisual corpus of 2.78 hours from 24 speakers, used as unseen test
  audio.

The system is intended as a foundation for Bangla virtual tutors, public-service
assistants, accessible interfaces and interactive media.

## 1.6 Organization of the Report

- **Chapter 2 — Background** covers the concepts needed for the rest of the
  report, reviews audio-driven talking-head generation, and identifies the gap for
  Bangla.
- **Chapter 3 — Project Design** presents the requirements, the system
  architecture, how the avatar model was selected and adapted, the gaps found
  in it, and the project plan.
- **Chapter 4 — Implementation and Results** describes the implementation and
  reports the effect of the fixes, Bangla phoneme-level results, the comparison
  with published systems, the leakage found in them, and speed and response time.
- **Chapter 5 — Standards and Design Constraints** covers the engineering
  standards, design constraints, cost, and the complex engineering problem
  mapping.
- **Chapter 6 — Conclusion** summarises what was delivered against what was
  proposed, the limitations, the journal paper in progress, and future work.

---

# Chapter 2: Background

This chapter introduces the concepts used in the rest of the report, reviews the
literature on audio-driven talking-head generation and similar applications, and
identifies the gap this project addresses.

## 2.1 Preliminaries

**Audio-driven talking-head generation.** Generating video of a face whose lip
movements follow a given speech signal.

**Person-specific and person-generic models.** A person-specific model is trained
for one face and must be retrained for each new person; a person-generic model
works on any face without retraining, usually at a higher computational cost.

**U-Net inpainting.** An encoder–decoder network with skip connections. In
SyncTalk_2D it receives a video frame with the mouth region blacked out and
"paints" the mouth back in, conditioned on the audio.

**Audio features.** Speech is converted into a sequence of numerical features
before it reaches the network — for example a mel spectrogram, the output of an
audio–visual encoder, or the internal representation of a self-supervised speech
model.

**Self-supervised speech encoders (wav2vec 2.0, XLS-R).** Models pre-trained on
large amounts of unlabelled speech [25]. XLS-R [13] is pre-trained on 128
languages, including Bangla, and its intermediate layers capture phonetic content.

**SyncNet and LSE.** SyncNet [11] is a network that judges whether a mouth and an
audio clip are in sync. From it come the two standard lip-sync metrics: LSE-D
(lip-sync error distance; lower is better) and LSE-C (lip-sync error confidence;
higher is better).

**Phoneme and viseme.** A phoneme is a distinct speech sound; a viseme is a
visually distinct mouth shape. Several phonemes can share a viseme, so a
phoneme-to-viseme mapping is many-to-one.

**Leakage.** Mouth motion in a generated video that comes from the input video
rather than from the audio. A model that leaks can look well synchronised while
ignoring the speech. It is measured by driving the model with silent audio and
counting the frames in which the mouth is open [17].

**Speech-to-speech conversational model.** A large language model that takes
speech in and produces speech out directly, instead of passing through separate
speech-recognition and speech-synthesis stages.

**Voice activity detection (VAD).** Detecting when a person is speaking, used to
decide when the user has finished their turn.

**Real-time communication (RTC).** Low-latency streaming of audio and video
between browser and server, provided here by LiveKit over WebRTC.

**Neural radiance fields (NeRF) and 3D Gaussian Splatting (3DGS).** 3D scene
representations used by several recent talking-head systems. NeRF represents the
head as a continuous neural field; 3DGS represents it as many small 3D Gaussian
"splats", which render much faster.

## 2.2 Literature Review

Speech-driven talking-head synthesis has developed in stages: from 2D
image-based animation, to 3D neural representations, to fast Gaussian-Splatting
rendering. Each stage addressed the weaknesses of the one before and introduced
new trade-offs.

**2D animation.** Early work animated a single static image. MakeItTalk [1]
separated the speech content from the speaker's identity to drive facial
landmarks and head motion from one image, and produced convincing lip-sync and
expression control. As a 2D method it did not represent 3D geometry, so head
rotation and depth-dependent motion were not consistent.

**Neural radiance fields.** To obtain 3D consistency, later work conditioned
Neural Radiance Fields on audio. AD-NeRF [2] was the first to do so, producing
3D-consistent, high-fidelity talking faces and upper bodies from new viewpoints.
It tied facial motion closely to the radiance field, however, which gave little
control over individual facial attributes. DFA-NeRF [3] disentangled lip motion
from head pose and blinking for more personalised and controllable generation.
NeRF-based methods nonetheless remained slow to train and to render, which made
real-time use impractical.

**Synchronisation.** SyncTalk [4] made synchronisation the central problem,
aligning audio, lip movement, facial expression and head pose through a
face-sync controller and stabilising mechanisms. It clearly improved temporal
coherence and realism over earlier NeRF models, but kept NeRF's computational
cost.

**Gaussian Splatting.** Recent work replaced NeRF with explicit 3D Gaussian
Splatting to remove the computational bottleneck. GaussianTalker [5] rendered
audio-driven 3D Gaussian talking heads in real time with high visual quality.
TalkingGaussian [6] kept the static facial structure separate from the
audio-driven deformation, reducing distortion in highly dynamic regions such as
the mouth. GaussianSpeech [7] added finer explicit modelling of the teeth and
inner mouth, and NeRFFaceSpeech [8] generated 3D-aware animation from a single
image using generative priors. InsTaG [9] combined Gaussian Splatting with a
universal motion field learned in pre-training, so that a new person can be
learned from a few seconds of video.

**2D lip-synchronisation.** In parallel, 2D methods edit only the mouth region of
an existing video. Wav2Lip [10] trained a generator against a pre-trained SyncNet
"lip-sync expert" [11] and became the most widely used baseline. IP-LAP [16] adds
landmark and appearance priors to preserve identity, and MuseTalk [15] and
LatentSync [14] generate the mouth in the latent space of an image or diffusion
model. SyncTalk_2D [18] is a lightweight person-specific 2D U-Net designed for
real-time use, and is the model this project builds on.

**Leakage.** Because 2D methods see the rest of the face, they can learn to copy
the mouth from the input video instead of generating it from the audio.
LatentSync [14] describes this as a "shortcut" that its training has to counter,
and KeySync [17] proposed a masking strategy against it together with a leakage
measure, LipLeak, which drives the model with silent audio. We use the same silent
audio test, and extend it in Section 4.3.4.

**The gap.** All of these systems are built and tested on high-resource
languages, chiefly English, and give little attention to the phonetic and prosodic
properties of low-resource languages such as Bangla. The most visually capable
methods also need GPUs far beyond consumer hardware. Few report whether the mouth
stays still in silence, and almost none include real video as a reference row in
their lip-sync tables. Our work focuses on a Bangla talking head that runs in real
time on a laptop, with its audio dependence verified directly.

### 2.2.1 Similar Applications

**(i) Tavus (2024–2025).** A commercial platform that generates personalised
AI-driven talking avatars for sales and marketing. The system is proprietary,
depends on large datasets and cloud-scale compute, and is designed for
English-speaking markets. Our system targets Bangla and runs its avatar locally
on consumer hardware.

> **[Figure 2.1: Tavus Interface — insert image]**

**(ii) EMO [24].** Generates expressive facial animation from audio using 2D
diffusion models, which gives very expressive results but at high computational
cost and without real-time operation. Our system trades some expressiveness —
it animates the mouth region only — for real-time rendering on a laptop.

> **[Figure 2.2: EMO Methodology — insert image]**

**(iii) SadTalker [23].** Predicts 3D morphable-model parameters from audio and
re-renders the face with a neural renderer. It is widely used but suffers from
identity drift, unstable eye regions and jittery motion. Because our system edits
only the mouth of a real recording of the speaker, the speaker's identity, eyes
and head motion come directly from real video.

> **[Figure 2.3: SadTalker — insert image]**

**(iv) InsTaG [9].** A 3D Gaussian Splatting talking head that combines a
universal motion field with a dynamic deformation network, giving strong identity
preservation and stable motion. It is trained on English audio. We initially
considered InsTaG for this project, but chose a 2D model because our requirement
was real-time rendering on a 4 GB laptop GPU.

> **[Figure 2.4: InsTaG Methodology — insert image]**

**(v) Wav2Lip and SyncNet-based systems.** Wav2Lip [10] generates the mouth in 2D
and is known for strong lip-sync scores. We found that those scores need care: on
our Bangla clip and on the public HDTF dataset, Wav2Lip scores *above real video*
of the same speaker on the standard lip-sync metric (Section 4.3.4), because it
was trained against the same SyncNet that grades it.

> **[Figure 2.5: Wav2Lip Interface — insert image]**

**(vi) NeRF-based head avatars.** Models such as AD-NeRF [2] give high-quality
reenactment but need long training and struggle with head poses outside the
training data. Gaussian Splatting models render faster, but still need more GPU
than a consumer laptop provides.

## 2.3 Gap Analysis

**Table 2.1: Comparison of existing talking-head synthesis methods with the
proposed system**

| | SyncTalk | GaussianTalker | InsTaG | TalkingGaussian | NeRFFaceSpeech | **Alapon (ours)** |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| Speech-driven talking-head generation | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Trained and evaluated on Bangla speech | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Bangla-specific audio adaptation | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Bangla audiovisual dataset | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Low-latency optimisation | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ |
| Real-time shown on a laptop GPU\* | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Silent-audio leakage test reported\* | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Real video scored as a reference\* | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Phoneme-level (viseme) supervision | ✗ | ✗ | ✗ | ✗ | ✗ | partial† |

\* As reported in the original publications, which evaluate on desktop or
datacentre GPUs.
† A Bangla phoneme-to-viseme mapping is defined (Section 3.2) and used for
evaluation; using it as a training signal is future work.

Current talking-head systems achieve speech-driven animation with high visual
quality, but none is trained or evaluated on Bangla, and none of those in
Table 2.1 reports whether its mouth stays still when the audio is silent — the
simplest test that the mouth is actually driven by the audio. Most also require
GPUs well beyond consumer hardware. The proposed system addresses these gaps with
Bangla training data, a Bangla-trained lip-sync expert, a silent-audio leakage
test with real video as the reference, and a model small enough to run in real
time on a laptop. Phoneme-level lip-shape supervision is only partly addressed and
is identified as future work.

## 2.4 Summary

Current talking-head systems such as SyncTalk, InsTaG, GaussianTalker and
NeRFFaceSpeech have made major advances in high-quality audio-driven facial
synthesis using NeRF and 3D Gaussian Splatting, reaching impressive realism and, in
some cases, real-time rendering on powerful GPUs. 2D lip-synchronisation methods
such as Wav2Lip, MuseTalk and LatentSync are widely used for dubbing. Almost all of
them, however, are developed and evaluated on high-resource languages, and rarely
check whether their mouth motion is truly driven by the audio. This motivates a
real-time Bangla speech-driven avatar that is trained on Bangla data, verified to
be driven by the audio rather than by the input video, and light enough to run on
consumer hardware.

---

# Chapter 3: Project Design

This chapter presents the requirements of the system, its architecture, how the
avatar model was selected, adapted to Bangla and improved, and the project plan.

## 3.1 Requirement Analysis

The system has two very different computational profiles. *Training* the avatar
model is done once per avatar and needs a datacentre-class GPU; we used Google
Colab (NVIDIA A100 and L4). *Running* the system must fit on consumer hardware:
the avatar renderer runs locally on a laptop NVIDIA RTX 3050 with 4 GB of memory,
while conversational reasoning is handled by a cloud API. The project also
requires Bangla audiovisual data for training and evaluation, and enough local
storage for video frames, landmarks and audio features.

### 3.1.1 Functional and Nonfunctional Requirements

**Functional Requirements**

1. **Interface:** The system captures the user's microphone audio in a web browser
   and displays the avatar's video and audio reply, with a live transcript.
2. **Secured Real-Time Communication Layer (RTC):** Audio and video are streamed
   between the user and the system over LiveKit, with access tokens issued by a
   token server.
3. **Conversational Processing Module:** The agent detects the end of the user's
   turn, understands the Bangla request, and generates a spoken Bangla reply that
   keeps the conversation history.
4. **Speech Output:** The reply is produced directly as Bangla speech by the
   speech-to-speech conversational model.
5. **Talking-Head Video Generation and Streaming:** The avatar server turns the
   reply audio into lip-synchronised video frames, streams them in step with the
   audio, and plays an idle animation when the avatar is not speaking.

**Non-Functional Requirements**

1. **Availability:** The system is accessible whenever the server is running, with
   minimal downtime.
2. **Security:** Users are authenticated through session tokens.
3. **Performance and Quality:** The system answers within 1–2 seconds of the user
   finishing speaking, and the avatar renders at no less than 25 frames per second.
4. **Reliability:** The system runs on a consumer laptop GPU without exhausting its
   memory.
5. **Linguistic Robustness:** Bangla speech is transcribed in Bangla script and
   answered in natural, formal-register Bangla.
6. **Modularity and Scalability:** The conversational agent, the real-time media
   layer and the avatar renderer are decoupled, so each can be upgraded or replaced
   independently.

### 3.1.2 System Diagram

Figure 3.1 gives a high-level overview of the system.

> **[Figure 3.1: System Diagram — redraw to match Section 3.2: Browser ↔ LiveKit ↔
> Agent (Gemini speech-to-speech) ↔ Avatar Server (Alapon)]**

### 3.1.3 Sequence Diagram

Figure 3.2 shows the order of interactions between the components during one
conversational turn.

> **[Figure 3.2: Sequence Diagram of the Proposed System — redraw: user speaks →
> VAD detects end of turn → Gemini generates spoken reply → reply audio sent to
> avatar server → frames and audio streamed back → browser plays them together]**

### 3.1.4 UI Design

> **[Figure 3.3: User Interface of our System — insert image]**
>
> **[Figure 3.4: User Interface of our System — insert image]**

## 3.2 Detailed Methodology and Design

### Overall System Architecture

The system is a voice-to-voice-and-video pipeline:

> Speech → Conversational model → Speech → Audio features → Lip-synchronised video

The user speaks Bangla in the browser. The audio reaches the agent over LiveKit.
The agent's conversational model produces a spoken Bangla reply, which is sent to
the avatar server. The avatar server converts the reply audio into
lip-synchronised video frames and streams them back, and the agent publishes the
audio and video together to the user through LiveKit. The components are
independent processes connected by LiveKit and WebSockets, so each can be
developed, debugged and replaced separately.

```
Browser ──LiveKit──► Agent (Gemini 3.1 Flash Live, speech-to-speech; Silero VAD)
                        │  reply audio, 24 kHz PCM, over WebSocket
                        ▼
                     Avatar server (streaming features → Alapon → JPEG frames)
                        │  frames + audio, segment protocol
                        ▼
Browser ◄──LiveKit── Agent publishes the audio and video tracks together
```

> **[Figure 3.5: System Architecture — redraw to match the diagram above]**

**How the design evolved.** The FYDP-1 design was a cascade of three separate
services: Whisper speech recognition through a Gradio API, Gemini 2.5 Flash for
text reasoning, and Microsoft Edge TTS (`bn-IN-TanishaaNeural`) for speech
synthesis. Each stage added its own latency, so we moved to Gemini's
speech-to-speech "native audio" model, which removed the separate recognition and
synthesis stages. That model, however, slowed down as a conversation went on:
its reply time grew from 8 seconds to 66 seconds within five turns. Switching to
`gemini-3.1-flash-live-preview` kept the reply time constant, and this is the
model used in the final system.

### Real-Time Communication Layer and Agent Control

The agent is built on the LiveKit Agents framework [20]. It creates the LiveKit
session, greets the user, and then listens continuously, routing the user's audio
to the conversational model and the reply to the avatar server. It manages the
session lifecycle and user interruptions, so the interaction behaves like a
conversation rather than a request–response exchange. A small Flask server issues
the LiveKit access tokens used by the browser.

### Bangla Conversational Agent

The conversational model [21] receives the user's speech directly and replies in
speech. Three Bangla-specific problems appeared that do not occur in English, and
each needed its own fix.

- **Wrong script in the transcript.** By default the model detected the language
  afresh on every utterance. With short turns of 0.6–1.8 seconds, phonetic overlap
  between Bangla, Hindi and Punjabi, and English loanwords, the same speaker's
  words came back in Devanagari, Gurmukhi and Roman script within a single session.
  We pinned the input language to `bn-BD`.
- **Turns cut off mid-question.** Bangla speakers pause mid-sentence, and the
  default end-of-turn detection answered half a question. We run a local
  voice-activity detector (Silero [19]) with the end-of-turn silence set to 0.55
  seconds, long enough to survive a natural Bangla pause.
- **Technical vocabulary pulled towards English.** Terms such as "মেশিন লার্নিং"
  and "ChatGPT" repeatedly shifted recognition away from Bangla. We supply them as
  biasing phrases.

The agent's persona, "Redwan", answers only in Bangla, in the formal register
(*আপনি*), and keeps established technical terms in their usual form.

### Avatar Rendering Server

The avatar server is a FastAPI application with WebSocket endpoints for audio in
and video out. At 25 frames per second it has a budget of 40 ms per frame.

- **Streaming audio features.** The model was trained on audio features extracted
  from whole recordings, but live audio arrives in chunks. Extracting features
  chunk by chunk shifts the timeline slightly at every boundary and slowly
  desynchronises the lips. Our streaming feature extractor produces the same
  features as offline extraction: it resamples with a fixed phase, recomputes mel
  frames from a hop-aligned tail with eight frames of context, and emits a feature
  only once all of its audio has arrived.
- **Segment protocol.** Each reply segment carries its frame count and its audio,
  so the client knows exactly how many frames to expect and plays them against the
  audio.
- **Audio never waits for video.** If a frame is late the previous frame is
  repeated, because a gap in the voice is far more noticeable than a repeated
  frame.
- **Cached source frames and idle animation.** Decoded video frames and their
  landmarks are cached in memory, and idle frames are pre-rendered, so the GPU is
  free to render speech when it arrives.

### Selecting the Avatar Model

The avatar model must render at least 25 frames per second on a 4 GB laptop GPU
while the rest of the system is running, and its lips must follow Bangla speech.
We tested the published lip-sync systems that have public code and weights on the
same 31-second Bangla clip, and compared them against these requirements
(Table 3.1). The lip-sync results for all candidates are in Section 4.3.3.

**Table 3.1: Candidate avatar models against the project requirements**

| Model | Type | Model size | Time to generate the 31 s clip (cloud T4 GPU) |
|---|---|--:|--:|
| **SyncTalk_2D** [18] | 2D, person-specific | **49 MB** | **2 min 33 s** |
| Wav2Lip [10] | 2D, person-generic | 436 MB | 9 min 06 s |
| IP-LAP [16] | 2D, person-generic | 475 MB | — |
| MuseTalk 1.5 [15] | 2D latent, person-generic | 3,400 MB | about 15 min |
| LatentSync 1.5 [14] | 2D latent diffusion, person-generic | 5,072 MB | about 30 min |
| InsTaG [9] | 3D Gaussian Splatting, person-specific | — | not run |

Size is the checkpoint file (IP-LAP: its two checkpoints together). The times are
for offline generation, which also reads and writes video files; "—" means not
timed.

SyncTalk_2D is 9 to 100 times smaller than the alternatives and generated the clip
3.6 to 12 times faster than those that were timed. On the laptop, in the live system with frames
kept in memory, it renders at 30 frames per second (Section 4.3.5). The person-
generic systems can animate any face without training, but the smallest of them
is still nine times larger, and none of those timed came near real time even on
the cloud GPU. InsTaG
was ruled out because 3D Gaussian Splatting training and rendering exceed the
4 GB laptop budget. SyncTalk_2D's cost is that it must be trained for each face —
about five minutes of video and seven hours of training — which is acceptable for
an avatar that represents one fixed speaker. Being person-specific also means the
speaker's identity, eyes and head motion come directly from real video; only the
mouth is generated.

### Alapon: Adapting SyncTalk_2D to Bangla

SyncTalk_2D [18] is a 2D U-Net that receives two images — a reference frame of
the speaker and the current frame with the mouth region blacked out — together
with audio features, and paints the mouth back in. **Alapon** is SyncTalk_2D
adapted to Bangla and improved in the following ways.

**1. Bangla training data.** The avatar is trained on 7,717 frames (about 5.1
minutes) of a Bangla speaker, `redwan`, at 25 fps and 1080×1080. The recording is
split into contiguous intervals: frames 0–6170 for training, 6171–6941 for
validation, and 6942–7713 (772 frames) for testing. Contiguous intervals are used
because neighbouring frames are almost identical, and a random split would put
near-copies of test frames into training.

**2. Bangla lip-sync expert.** During training, a SyncNet "expert" judges whether
the generated mouth matches the audio. We retrained the expert on the Bangla
speaker with contrastive sampling: half of all training pairs are mismatched, with
the audio taken at least five frames away from the video, so the expert must learn
to tell a correct mouth from a shifted one. The avatar was trained against this
Bangla expert, with the sync loss switched on from epoch 5 at a weight of 0.03.

**3. Streaming audio features** for the live system, described above.

**4. Multilingual audio encoder (XLS-R).** The default audio features come from an
audio–visual encoder trained on English. We also tried XLS-R [13], a speech
encoder pre-trained on 128 languages including Bangla, in place of these
features. It did not work (Section 4.3.6), so Alapon uses the default features.

**5. Closing the model's performance gaps.** While training the Bangla model we
found that its mouth moved even when the audio was silent, and its lips did not
close fully on প, ব, ম. Investigating this revealed four gaps in the SyncTalk_2D
model that held back its performance (Table 3.2). Alapon addresses all four.

**Table 3.2: Gaps in the SyncTalk_2D model and how Alapon addresses them**

| Gap in the model | Effect on performance | How Alapon addresses it |
|---|---|---|
| **The mouth mask does not cover the jaw.** The model is meant to see the mouth region fully hidden, but the bottom ten rows of the face crop — the chin and jaw — stay visible in training and in use. | The model reads mouth opening from the visible jaw instead of the audio, so the mouth moves during silence and lip closures are weak. With audio and reference held constant, this ten-pixel strip produced **91% of all mouth motion**. | Mask extended to cover the jaw (Section 4.3.1). |
| **The lip-sync expert learns only from matching audio–video pairs.** | It never sees an out-of-sync example, so it approves every mouth and gives no useful training signal — consistent with the FYDP-1 observation that SyncNet scores stayed near 1.0 at every offset. | Expert retrained with mismatched pairs (item 2). |
| **The lip-sync loss is weighted 10×.** | Together with the gap above, a strong but meaningless training signal that competes with image quality. | Weight 0.03, switched on from epoch 5. |
| **Training and use see different reference frames.** | A mismatch between what the model learns and what it sees at run time. | One fixed reference frame in both. A control run showed this changes leakage by only 1%; the mask is the whole mechanism. |

**6. A clean training split.** While preparing the final training runs we also
found that training had drawn frames, including reference frames, from the whole
video, test interval included. Training now uses only the training interval, and
the interval is recorded with each trained model. The
models evaluated in this report were trained before this fix; Section 4.2 explains
which results this affects.

**7. Bangla phoneme-to-viseme mapping.** A *viseme* is a visually distinct mouth
shape; several phonemes can share one. Table 3.3 maps the Bangla sound inventory
to eleven visemes.

**Table 3.3: Bangla phoneme-to-viseme mapping**

| Viseme | Mouth shape | Bangla letters | Phonemes (IPA) |
|---|---|---|---|
| V0 | Rest, lips closed | (silence, pause) | — |
| V1 | Lips pressed together | প ফ ব ভ ম | /p pʰ b bʱ m/ |
| V2 | Lips apart, neutral; tongue at teeth or ridge | ত থ দ ধ ট ঠ ড ঢ ড় ঢ় ন ণ ল র স ৎ | /t̪ t̪ʰ d̪ d̪ʱ ʈ ʈʰ ɖ ɖʱ ɽ ɽʱ n l r s/ |
| V3 | Lips slightly protruded | চ ছ জ ঝ য শ ষ | /tʃ tʃʰ dʒ dʒʱ ʃ/ |
| V4 | Neutral; jaw follows the vowel | ক খ গ ঘ ঙ ং হ | /k kʰ g gʱ ŋ ɦ/ |
| V5 | Spread, close | ই ঈ | /i/ |
| V6 | Spread, mid-open | এ, অ্যা | /e æ/ |
| V7 | Wide open, jaw dropped | আ | /a/ |
| V8 | Open, slightly rounded | অ | /ɔ/ |
| V9 | Rounded, mid | ও | /o/ |
| V10 | Rounded and protruded, close | উ ঊ | /u/ |

**[TODO: have a team member confident in Bangla phonetics check this table]**

The mapping makes three properties of Bangla explicit. Aspiration (ক/খ, ত/থ, দ/ধ)
does not change lip shape, so each aspirated consonant shares a viseme with its
unaspirated pair. Dental and retroflex consonants (ত/ট, দ/ড) differ in tongue
position rather than lip shape and also share a viseme. Vowel length in spelling
(ই/ঈ, উ/ঊ) and nasalisation (চন্দ্রবিন্দু) are not visible on the lips, and
diphthongs (ঐ, ঔ) are sequences of two vowel visemes. These distinctions must
therefore be carried by the audio, not by the mouth shape. The mapping defines the
classes for the phoneme-level evaluation in Section 4.3.2; using it as a training
signal is future work.

### Bangla Audiovisual Corpus

We also collected a corpus of 2.78 hours of Bangla speech video from 24 speakers
(28 videos). It was not used to train Alapon, which is person-specific and uses a
single speaker; six of its speakers are used as unseen test audio in
Section 4.3.2.

### Performance and System Configuration

The system runs in a hybrid configuration: the avatar renderer, the agent and the
browser client run locally, and conversational reasoning is handled by the Gemini
API. The runtime machine has an AMD Ryzen 7 5800H processor, 8 GB of RAM, and an
NVIDIA RTX 3050 Laptop GPU with 4 GB of memory. Training ran on Google Colab A100
and L4 GPUs; the benchmark of published systems ran on Colab A100, L4 and T4 GPUs.
Measured speed and latency are reported in Section 4.3.5.

## 3.3 Project Plan

The project was carried out as a staged development with parallel work, iterative
validation and continuous documentation.

The first phase defined the problem, analysed the requirements and reviewed the
literature on Bangla speech processing, real-time voice assistants and audio-driven
lip synchronisation. It established the technical basis and feasibility of the
system.

> **[Figure 3.6: Gantt Chart for FYDP-1 — insert image]**

The second phase designed the architecture and built the English baseline: the
voice pipeline, real-time streaming over LiveKit, and the integration of the
avatar renderer. Modules were built and tested separately before being combined,
and continuous testing checked real-time performance, latency and stability.

> **[Figure 3.7: Gantt Chart for FYDP-2 — insert image]**

The third phase moved the system to Bangla and built Alapon: the Bangla agent, the
selection of the avatar model, training on Bangla video, the Bangla lip-sync
expert, closing the model's performance gaps, a trial of the XLS-R encoder, the
benchmark of published systems on Bangla and HDTF video, and the Bangla
phoneme-level evaluation. The final step was the preparation of this report and
the start of the journal paper.

> **[Figure 3.8: Gantt Chart for FYDP-3 — insert image]**

## 3.4 Task Allocation

The work was divided among the six team members so that development of the
pipeline, research and experiments, and documentation could proceed in parallel.

**Table 3.4: Distribution of Project Responsibilities**

| Team Member | Primary Role | Key Responsibilities and Deliverables |
|---|---|---|
| Member 1 | Systems Architect + R&D | Integration of the conversational agent pipeline; LiveKit infrastructure; system-level latency and performance. |
| Member 2 | R&D and Creative | Bangla speech and lip-sync research; comparative analysis; project imagery and posters. |
| Member 3 | Documentation & Lead Coordinator | Report structure; project coordination; quality control of academic content and consistency. |
| Member 4 | Module Developer | Implementation of system modules; data preparation for model training; unit testing. |
| Member 5 | Validation Engineer | Debugging; performance monitoring; iterative refinement of system responses. |
| Member 6 | Implementation Support | UI/UX components; validation of software deliverables; integration testing. |

## 3.5 Summary

This chapter set out the requirements and design of the real-time Bangla
conversational avatar. The requirements separate a one-off training workload,
which uses cloud GPUs, from a runtime workload that must fit on a consumer laptop.
The architecture connects a browser client, a Bangla speech-to-speech agent and an
avatar rendering server through LiveKit and WebSockets. SyncTalk_2D was selected
as the avatar model after testing it against published alternatives, because it is
the only candidate small and fast enough for a laptop. Building Alapon from it
involved Bangla training data, a Bangla lip-sync expert, streaming audio features,
closing four gaps in the model's design and training, a clean training split,
and a Bangla
phoneme-to-viseme mapping; a multilingual XLS-R encoder was also tried but did not
work. The project plan and task allocation describe how the six-member team
divided this work. Chapter 4 reports the implementation and results.

---

# Chapter 4: Implementation and Results

This chapter describes the implementation, how the system was tested, and the
results.

## 4.1 Environment Setup

The system has three modules:

- **agent** — the LiveKit Agents application running the Bangla conversational
  agent on Gemini 3.1 Flash Live, with local voice-activity detection. An English
  agent is kept as a baseline.
- **frontend** — a Flask token server and a browser client showing the avatar video
  and a live transcript.
- **SyncTalk_2D** — Alapon, the avatar model, and a FastAPI server that turns reply
  audio into video frames.

A fourth module, **benchmark**, holds the scoring code used for every system in
Section 4.3, so that all systems are measured by the same code.

Software: Python 3.10, PyTorch 2.2 (CUDA), OpenCV, FFmpeg, Librosa, FastAPI,
LiveKit Agents and the Gemini API. Training and benchmarking used Google Colab
GPUs. All speed measurements were made on the target laptop: AMD Ryzen 7 5800H,
8 GB RAM, NVIDIA RTX 3050 Laptop GPU (4 GB).

## 4.2 Testing and Evaluation

**Test data.**

- **Bangla test clip:** the last 772 frames (31 seconds) of the Bangla recording.
  The published systems never saw this video. Alapon was trained before we found
  that the training data included these frames (Section 3.2, item 6), so its
  results on this clip — lip-sync, reconstruction and phonemes — may be somewhat
  optimistic. The silence test is not affected, because the extended mask hides
  the pixels the model would copy. The six-speaker test in Section 4.3.2 is also
  not affected.
- **Six new Bangla voices:** audio from six speakers in our corpus that Alapon had
  never heard, used to drive the avatar's face.
- **HDTF [12]:** a public English talking-head dataset; six speakers, one
  31-second clip each, used to check the published systems on a second dataset.

**Metrics.**

| Metric | What it measures | Better |
|---|---|---|
| LSE-D | Lip-sync distance between mouth and audio | Lower |
| LSE-C | Lip-sync confidence | Close to the real-video score |
| Mouth moving during silence | % of frames with the mouth open when the model is driven by silent audio | Lower |
| Articulation | How often the mouth opens during speech, relative to the real speaker (1.00 = same) | Close to 1.00 |
| Open ÷ closed | Mouth opening on আ অ divided by mouth opening on প ফ ব ভ ম | Close to the real speaker |
| PSNR, SSIM, MAE | How closely the generated face matches the real one | Higher, higher, lower |
| Speed, GPU memory | Frames per second and memory on the laptop | Higher / lower |

Lip-sync is scored with the original SyncNet [11], which is independent of the
model we trained. **The real video is scored too, as a reference:** a real person
is perfectly in sync with their own voice, so a system should come *close to* the
real-video score, not above it. The mouth counts as open when the gap between the
inner lips exceeds 3% of the face width.

**Compared systems.** Four published methods, six checkpoints: Wav2Lip [10] (GAN
and non-GAN checkpoints), IP-LAP [16], MuseTalk 1.0 and 1.5 [15], and
LatentSync 1.5 [14]. We ran all of them ourselves, on the same clips, with the same
scoring code. SyncTalk_2D before the fix — trained by us on the same Bangla data
with the original mask — is included as a control.

## 4.3 Results and Discussion

### 4.3.1 Fixing the mouth mask

**Table 4.1: SyncTalk_2D before and after the mouth-mask fix (Bangla test clip)**

| | Before fix | After fix (Alapon) |
|---|:-:|:-:|
| Mouth moving during silence ↓ | 41% | **0.1%** |
| PSNR ↑ | 29.8 dB | **33.1 dB** |
| SSIM ↑ | 0.88 | **0.91** |
| MAE ↓ | 0.021 | **0.014** |
| LSE-C (real video: 5.14) | 5.16 | 5.11 |
| Articulation (real video: 1.00) | 1.05 | 0.94 |

Before the fix, the avatar's mouth moved in 41% of frames even when there was no
sound at all. The real speaker opens their mouth in about half of all frames, so
the model was reproducing about 80% of the real mouth movement without any audio —
it was copying the visible jaw instead of listening. After the fix this drops to
0.1%, and reconstruction improves on all three measures. The mouth still moves
normally with speech: it opens 94% as often as the real speaker's.

The LSE-C row shows why the silence test is needed: the lip-sync score is almost
the same before and after the fix (5.16 against 5.11), so the standard metric
alone could not have revealed the problem.

To confirm the cause, we trained six identical models that differ only in how many
rows of the jaw stay visible (Table 4.2).

**Table 4.2: Visible jaw rows against mouth movement during silence**

| Visible jaw rows | 10 (original) | 8 | 6 | 4 | 2 | **0 (fixed)** |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| Mouth moving during silence | 39% | 36% | 26% | 33% | 36% | **8%** |
| LSE-C | 5.07 | 5.09 | 4.88 | 4.98 | 5.14 | 4.93 |

Only covering the jaw completely stops the copying; leaving even two rows visible
does not. Across all six models the LSE-C score barely changes and shows no trend.
These six models were trained without the lip-sync loss; Alapon, which adds it,
reaches 0.1%.

### 4.3.2 Bangla phoneme-level evaluation

To check that the mouth follows Bangla *sounds*, not just loudness, we labelled
every video frame with the Bangla sound being spoken and measured how far the mouth
is open for each group of sounds in Table 3.3. A Bangla speech-recognition model
[22] marks which Bangla letter is spoken at each moment. The timing was aligned
once, on the real video — the mouth moves 80 ms before the model marks the sound —
and the same rule was applied to every system. Mouth opening is the gap between the
inner lips divided by the width of the face.

**Table 4.3: Mouth opening for closed and open Bangla sounds (test clip)**

| System | Closed sounds (প ফ ব ভ ম) | Open sounds (আ অ) | Open ÷ closed |
|---|:-:|:-:|:-:|
| Real video | 0.011 | 0.045 | 4.0× |
| **Alapon (ours)** | **0.011** | **0.043** | **3.8×** |
| SyncTalk_2D, before fix | 0.015 | 0.045 | 3.0× |
| Wav2Lip (GAN) | 0.018 | 0.053 | 3.0× |
| Wav2Lip (non-GAN) | 0.014 | 0.041 | 3.0× |
| LatentSync 1.5 | 0.010 | 0.031 | 3.1× |
| MuseTalk 1.5 | 0.009 | 0.025 | 2.8× |
| MuseTalk 1.0 | 0.005 | 0.018 | 3.5× |
| IP-LAP | 0.016 | 0.029 | 1.9× |

- **Alapon has the contrast closest to the real speaker** (3.8× against 4.0×).
  Its lips close on প, ব, ম exactly as far as the real speaker's (0.011) and open
  on আ almost as wide (0.043 against 0.045).
- **Before the fix**, the model did not close its lips fully on প, ব, ম (0.015),
  because a lip closure does not show in the jaw it was copying. The fix raised the
  contrast from 3.0× to 3.8×.
- **Wav2Lip** opens too wide on আ (GAN: 0.053) and does not fully close on প, ব, ম.
  **LatentSync** closes correctly but opens too little on আ (0.031).
- **MuseTalk and IP-LAP** barely move: MuseTalk 1.0's ratio looks reasonable only
  because both of its numbers are tiny.

The published systems never saw this video, so the test is fair to them. Alapon's
row shares the caveat in Section 4.2, so we also tested it on voices it had never
heard: six Bangla speakers from our corpus. In this test the face is still our
speaker's, but the audio comes from someone else, so the mouth can only follow the
audio.

**Table 4.4: Open ÷ closed with six new Bangla speakers**

| Speaker | Alapon | Before fix |
|---|:-:|:-:|
| 02 | **2.55×** | 1.99× |
| 04 | **1.96×** | 1.49× |
| 08 | **2.01×** | 1.82× |
| 11 | **1.88×** | 1.63× |
| 23 | **3.18×** | 1.76× |
| 24 | **2.76×** | 1.59× |
| **All six** | **2.3×** | **1.7×** |

Alapon keeps a clear contrast between open and closed sounds with new voices, and
is higher than the model before the fix for **every one of the six speakers**. The
model before the fix copies the mouth of the original video, which is saying
different words, so its mouth separates the sounds less. The contrast is smaller
than the real speaker's (4.0×), so Alapon follows new voices less strongly than
the voice it was trained on.

We also measured lip width to separate rounded sounds (উ, ও) from spread ones
(ই, এ), but it did not show a clear difference between the systems, so we do not
claim that the mouth *shape* matches each sound.

### 4.3.3 Comparison with published systems

**Table 4.5: Comparison on the Bangla test clip**

| System | Size | LSE-D ↓ | LSE-C | Mouth moving during silence ↓ | Articulation |
|---|--:|:-:|:-:|:-:|:-:|
| Real video | — | 7.39 | 5.14 | — | 1.00 |
| **Alapon (ours)** | **49 MB** | 7.30 | 5.11 | **0.1%** | 0.94 |
| SyncTalk_2D, before fix | 49 MB | 7.32 | 5.16 | 41% | 1.05 |
| Wav2Lip (GAN) | 436 MB | 7.26 | 6.08 | 46% | 1.21 |
| Wav2Lip (non-GAN) | 436 MB | 6.83 | 6.44 | 0.8% | 1.05 |
| LatentSync 1.5 | 5,072 MB | 7.29 | 5.09 | 36% | 0.79 |
| MuseTalk 1.5 | 3,400 MB | 7.86 | 4.90 | 0.4% | 0.53 |
| MuseTalk 1.0 | 3,400 MB | 7.72 | 4.67 | 0.1% | 0.22 |
| IP-LAP | 475 MB | 8.71 | 4.36 | 20% | 0.76 |

Real video is not a model, so it has no size and cannot be driven by silent audio.

- **Alapon is closer to the real-video lip-sync score** (5.11 against 5.14) than
  any published system, and has the least mouth movement during silence (0.1%,
  level with MuseTalk 1.0) while still moving normally during speech
  (articulation 0.94).
- **Both Wav2Lip checkpoints score above real video** (6.08 and 6.44 against
  5.14). This is not better lip-sync: Wav2Lip was trained against the same SyncNet
  that scores it.
- **LatentSync** is close to real on lip-sync (5.09), but is 100 times larger than
  Alapon and its mouth moves during silence in 36% of frames.
- **MuseTalk** barely moves during silence, but it also barely moves during speech
  (articulation 0.53 and 0.22): it avoids mistakes by barely moving at all.

### 4.3.4 Leakage in published systems

Having found that SyncTalk_2D copied the mouth from its input video, we asked
whether the published systems do the same. We drove all six published checkpoints
with silent audio on the Bangla clip and on six HDTF speakers, and scored them
with the same code as Alapon.

**Table 4.6: Published systems on the public English dataset HDTF (six speakers,
averages)**

| System | LSE-D ↓ | LSE-C | Above real video? | Mouth moving during silence ↓ | Articulation |
|---|:-:|:-:|:-:|:-:|:-:|
| Real video | 7.42 | 8.01 | — | — | 1.00 |
| Wav2Lip (GAN) | 6.93 | 8.75 | **yes** | 24% | 1.03 |
| Wav2Lip (non-GAN) | 6.69 | 8.94 | **yes** | 8% | 0.98 |
| LatentSync 1.5 | 6.70 | 8.82 | **yes** | 24% | 0.64 |
| MuseTalk 1.5 | 7.12 | 8.38 | **yes** | 46% | 0.99 |
| MuseTalk 1.0 | 7.72 | 7.52 | no | 41% | 0.75 |
| IP-LAP | 7.07 | 7.88 | no | 10% | 0.34 |

Alapon is not in this table: it is person-specific and would have to be trained
separately for each HDTF speaker.

Five findings follow from Tables 4.1, 4.2, 4.5 and 4.6.

1. **Five of the six published checkpoints leak on at least one dataset.** Their
   mouths moved during silence in 20–46% of frames on at least one of the two
   datasets. Only Wav2Lip (non-GAN) stayed low on both (0.8% on Bangla, 8% on
   HDTF — the level of our own jaw-covered models in Table 4.2). The gap we
   closed in SyncTalk_2D is not unique to it.
2. **The standard lip-sync metric scores generated video above real video.** On
   HDTF, four of six systems score above the real speakers on LSE-C. For three of
   them — Wav2Lip, Wav2Lip GAN and LatentSync — the difference is statistically
   significant across the six speakers (paired t-test, p < 0.05). Real video is
   the ceiling by definition, so a metric that ranks synthesis above it is not
   measuring what its name claims. Most published tables omit the real-video row,
   which hides this.
3. **The lip-sync metric cannot see leakage.** In Table 4.2, removing the visible
   jaw cut leakage from 39% to 8% while LSE-C stayed flat. We had expected the
   metric to *reward* leakage; the experiment did not support that, and we report
   the negative result. What it does show is that the metric is blind to it: the
   most-leaking system on HDTF, MuseTalk 1.5, still scores above real video.
4. **Measured leakage depends on the speaker.** A system can only copy the mouth
   movement that is in its input video, so speakers who move their mouths more
   reveal more leakage. Across all 36 speaker–system pairs on HDTF, leakage rises
   with the speaker's own mouth movement (rank correlation +0.64, p < 0.001). One
   HDTF speaker barely opens their mouth, and nearly every system scores close to
   zero leakage on that speaker. We therefore propose dividing leakage by how much the
   real speaker moves, so that results can be compared across speakers and
   datasets.
5. **One clip is not enough to judge a system.** MuseTalk 1.5 moved its mouth
   during silence in 0.4% of frames on the Bangla clip but 46% on HDTF, while
   LatentSync and Wav2Lip GAN leaked more on the Bangla clip than on HDTF. Leakage
   should be measured over several speakers and reported with its spread —
   including our own 0.1%, which comes from one Bangla clip. For Alapon the
   extended mask itself is the stronger evidence: the model cannot copy jaw pixels it can
   no longer see.

These findings go beyond the FYDP system and are the subject of the journal paper
described in Section 6.3.1.

### 4.3.5 Speed and response time

**Table 4.7: Performance on the laptop (RTX 3050, 4 GB)**

| | Result | Requirement |
|---|:-:|:-:|
| Model size | 49 MB | — |
| Rendering speed | 30 frames per second (33 ms per frame) | 25 fps |
| GPU memory used | 0.3 GB of 4 GB | — |
| Response time (user stops speaking → avatar answers) | 1.5–2 s | 1–2 s |

The avatar renders faster than the 25 frames per second it needs, using less than a
tenth of the laptop's GPU memory, and the live system sustains about 28 frames
per second once audio handling and streaming are added. The response time has
three parts: the end-of-turn wait (0.55 s, set deliberately so that natural
Bangla pauses are not cut off), Gemini's reply generation (about 0.5 s), and the
avatar pipeline from reply audio to moving lips (about 1.0 s). The parts overlap
because audio is streamed, giving 1.5–2 seconds in total, within the requirement.
The agent logs these timings live, and Gemini's generation time stays flat
across a long conversation.

### 4.3.6 Bangla audio encoder (XLS-R)

We also trained the model with a multilingual speech encoder, XLS-R [13], in place
of the default audio features, expecting it to represent Bangla sounds better. It
did not work. The first attempt used the encoder's last layer, whose output barely
changes from frame to frame, so the lip-sync expert could not learn at all. With a
middle layer (layer 12) it learned, but weakly: it separated in-sync from
out-of-sync audio about half as well as the expert trained on the default
features. The avatar trained against this weaker expert moved its mouth more
during silence than the default model, so Alapon uses the default audio features.
The default features come from an encoder pre-trained specifically for
audio–visual synchronisation, which is a strong starting point; a general speech
representation would need more than five minutes of a single speaker to learn the
mapping to mouth movement.

### 4.3.7 Discussion

1. **The mouth-mask fix was the key result.** It stopped the model copying mouth
   movement from the video (41% → 0.1%), improved reconstruction (PSNR 29.8 →
   33.1 dB), and sharpened Bangla sounds (open ÷ closed 3.0× → 3.8×).
2. **Alapon follows Bangla sounds.** Its lips close on প/ব/ম and open on আ closer
   to the real speaker than any other system, and still clearly for six Bangla
   speakers it had never heard (2.3× against 1.7× before the fix).
3. **Small and fast.** At 49 MB, Alapon matches the lip-sync of a 5 GB model and
   runs at 30 fps on a laptop.
4. **The leak is a field-wide problem.** Five of the six published systems we
   tested leak on at least one dataset, and the standard lip-sync metric neither detects leakage
   nor keeps generated video below real video. Evaluating a talking head needs the
   silence test and the real-video reference alongside the lip-sync score.
5. **Meets the requirements.** 30 fps against the 25 fps target, and a 1.5–2 s
   response against the 1–2 s target.

## 4.4 Summary

Alapon, the avatar model, is 49 MB, renders at 30 frames per second on a laptop
RTX 3050 using 0.3 GB of GPU memory, and the system answers within 1.5–2 seconds.
Fixing the mouth mask reduced mouth movement during silence from 41% to 0.1%,
raised PSNR from 29.8 to 33.1 dB, and raised the contrast between open and closed
Bangla sounds from 3.0× to 3.8×, against 4.0× for the real speaker. On the Bangla
test clip Alapon is closer to the real-video lip-sync score (5.11 against 5.14)
than any published system tested. For six Bangla speakers it had never heard, its
mouth opens 2.3 times wider on আ than on প/ব/ম, against 1.7 times before the fix.
The same silence test showed that five of the six published systems leak on at
least one dataset, and that the standard lip-sync metric scores three of them significantly
above real video. The multilingual XLS-R encoder did not work and was not used in
Alapon.

---

# Chapter 5: Standards and Design Constraints

This chapter relates the project to professional and ethical standards, describes
the constraints that shaped its design, analyses its cost, and shows that it
constitutes a complex engineering problem.

## 5.1 Compliance with the Standards

The system aligns with established professional and ethical standards, including
the ACM Code of Ethics and Professional Conduct, the ASME Code of Ethics and the
IEEE Code of Ethics. The following subsections set out the specific principles
that apply.

### 5.1.1 Software Standards

**ACM Code of Ethics — Principle 2.1: Strive to achieve high quality in both the
processes and products of professional work.** The system uses established models,
careful adaptation and systematic testing, and every result reported is measured
with a documented, repeatable procedure.

**ACM Code of Ethics — Principle 2.5: Give comprehensive and thorough evaluations
of computer systems and their impacts, including possible risks.** The system is
evaluated on lip-sync accuracy, mouth movement during silence, reconstruction
quality, Bangla sounds, speed and response time, against six published
checkpoints on two datasets, and its limitations are reported explicitly
(Section 6.2).

**ACM Code of Ethics — Principle 2.7: Foster public awareness and understanding
of computing, related technologies, and their consequences.** By documenting how
the avatar works, what it can and cannot do, and how talking-head metrics can
mislead, the project promotes understanding of these technologies and their
impacts.

**ACM Code of Professional Responsibilities — Code 2.2: Maintain high standards
of professional competence, conduct, and ethical practice.** The team followed
ethical practice throughout the project.

**ACM Code of Professional Responsibilities — Code 2.9: Design and implement
systems that are robustly and usably secure.** Access to the system is controlled
by session tokens, and the design aims to prevent misuse of the avatar and to
remain intuitive and safe to operate.

**IEEE Code of Ethics — Clause 3: Be honest and realistic in stating claims or
estimates based on available data.** Every performance figure in this report is
measured, and its source is recorded. Where an earlier estimate was not supported
by measurement — the end-to-end latency figure of the FYDP-1 design — it has been
replaced with measured values. Where a result has a known weakness, such as the
training data that included the test frames, or an expectation that the
experiments did not support, that is stated.

**IEEE Code of Ethics — Clause 6: Maintain and improve technical competence.**
The solution lies within the team's field of study, Computer Science and
Engineering.

**ASME Code — Fundamental Canon 2: Engineers shall perform services only in the
areas of their competence.** The project draws on machine learning, deep learning,
computer vision and real-time systems, all within the team's discipline.

### 5.1.2 Hardware Standards

**ASME Code of Ethics — Fundamental Canon 2: Engineers shall perform services
only in areas of their competence.** The project uses consumer GPUs and cloud GPU
resources that match the team's expertise in machine learning and neural
rendering.

**ASME Code of Ethics — Fundamental Canon 1: Engineers shall hold paramount the
safety, health, and welfare of the public.** Training runs on cloud GPUs and the
system runs on an ordinary laptop, so it poses no particular physical risk. The
project is not intended for safety-critical use without further validation.

**IEEE Code of Ethics — Clause 1: Hold paramount the safety, health, and welfare
of the public.** The hardware used is standard consumer and cloud computing
equipment and poses no direct physical risk to the public.

**IEEE Code of Ethics — Clause 6: Maintain and improve technical competence.**
Hardware choices are documented and measured — for example, the choice of a
laptop-class GPU as the runtime target — and the team worked within its technical
training.

### 5.1.3 Communication Standards

**ACM Code of Ethics — Principle 1.3: Be honest and trustworthy.** The methodology,
dataset limitations and model assumptions are documented in this report and in the
project's code, including results that did not support our initial expectations
(Sections 4.3.4 and 4.3.6).

**ACM Code of Ethics — Principle 2.3: Know and respect existing rules pertaining to
professional work.** The project follows applicable rules on data handling,
privacy and intellectual property. The Bangla corpus is used for research only and
is not redistributed until the licences of its source recordings have been traced
(Section 6.2).

**IEEE Code of Ethics — Clause 5: Improve the understanding of technology, its
appropriate application, and potential consequences.** The project provides
documentation and demonstration videos explaining how the system works, its
intended uses and its limitations, particularly for a low-resource language. The
performance gaps found in SyncTalk_2D and the weaknesses found in the
standard evaluation are being reported to the research community through the
journal paper.

### 5.1.4 Societal Standards

**ACM Code of Ethics — Principle 1.1: Contribute to society and to human
well-being.** The system serves Bangla speakers, who are underserved by current
AI technology, and is designed to be deployable where resources are limited.

**ACM Code of Ethics — Principle 1.6: Respect privacy.** The avatar is trained on
video of a speaker who consented to its use **[TODO: confirm — the `redwan`
recording is of team member Sheikh Redwanul Islam, recorded with his consent]**.
The system performs no identity recognition or surveillance. The user's speech is
sent to the Gemini API to generate replies, which users should be told.

**IEEE Code of Ethics — Clause 10: Assist colleagues and co-workers in their
professional development.** The project's source code and evaluation tools are
documented so that others can reproduce and build on the work. The Bangla corpus
will be released once the licences of its source recordings have been traced.

## 5.2 Design Constraints

The design of the Real-Time Bangla Speech-Driven Avatar Generation system was
shaped by several practical and contextual constraints.

### 5.2.1 Economic Constraint

High-fidelity talking-head models usually need large GPUs, large datasets and
cloud servers, which small institutions cannot afford. The system addresses this
directly: its avatar model, Alapon, is 49 MB, renders in real time on a consumer
laptop GPU using 0.3 GB of memory, and needs no cloud GPU at runtime. Training
uses modest cloud GPU time once per avatar. All tools and libraries are free or
open-source, and the training data is locally collected Bangla video.

### 5.2.2 Ethical Constraint

A system that generates realistic facial movement from speech carries risks of
impersonation, deepfakes and privacy violation. To limit these, the avatar is
trained only on footage of a speaker who consented, the system is intended for
controlled and transparent use, and it is not used to produce false or
unauthorised representations of real people. The Bangla corpus was assembled from
publicly available recordings for research; the speakers did not individually
consent, so the corpus is not redistributed, and its release depends on tracing
the source licences. Informed consent, anonymisation where necessary, and
responsible data processing guide any further data collection, including the
planned human evaluation.

### 5.2.3 Social Constraint

Language inclusivity and cultural representation are central. Current
talking-head systems focus on English, which disadvantages speakers of
low-resource languages such as Bangla. This system is built specifically for
Bangla users, with attention to Bangla phonetics, formal register and speech
rhythm. Bangla also varies by dialect in pronunciation, intonation and vocabulary,
which makes a single model representative of all speakers difficult to build from
limited data. The corpus includes 24 speakers, but it has not yet been audited
for balance across gender, region or speaking style.

## 5.3 Cost Analysis

The project was planned to be cost-effective and feasible in an academic setting.
Because it is almost entirely software-based, its financial requirements are low.

- **Hardware resources:** The system runs on a personal laptop. The avatar needs a
  CUDA-capable GPU, but a consumer laptop GPU is sufficient; no specialised
  hardware is required.
- **Software resources:** All development tools, programming languages, deep
  learning frameworks and audio–visual libraries are free or open-source, so there
  are no licensing costs.
- **Data and experimentation:** Training and evaluation use locally collected
  Bangla video and a public benchmark (HDTF), so no paid datasets are needed.
- **Operational costs:** The avatar runs locally. The conversational model is
  accessed through the Gemini API.

**Table 5.1: Detailed cost**

| Environment | Component | Used | Price |
|---|---|---|---|
| Local | CPU | AMD Ryzen 7 5800H | Owned |
| Local | GPU | NVIDIA RTX 3050 Laptop, 4 GB | Owned |
| Local | RAM | 8 GB | Owned |
| Cloud | GPU (training and benchmark) | Google Colab Pro — A100, L4, T4 | **[TODO: actual Colab spend]** |
| Cloud | GPU | Kaggle T4 | Free |
| Cloud | Storage | Google Drive | Free |
| Cloud | Conversational model | Google Gemini API | **[TODO: free tier or actual cost]** |

The cost analysis confirms that the project can be completed with very little
expenditure, which makes it appropriate and sustainable for an undergraduate final
year design project.

## 5.4 Complex Engineering Problem

### 5.4.1 Complex Problem Solving

Whether a problem is a complex engineering problem is determined by whether it
satisfies the attributes P1 to P7.

**P1: Depth of Knowledge Required.** P1 is satisfied when the project requires one
or more of the knowledge profiles K3, K4, K5, K6 or K8.

- **K1: Engineering Fundamentals.** The project uses mathematics, signal processing
  and computer science: audio is represented and transformed mathematically, and
  engineering principles govern real-time synchronisation and control.
- **K2: Discipline-Specific Knowledge.** Core computer science applies — machine
  learning, computer vision, data structures, algorithms and software
  architecture — with frame-based video rendering and audio processing as key
  discipline-specific elements.
- **K3: Specialist Knowledge.** Audio-to-visual speech synthesis requires specialist
  knowledge of deep learning. Lip synchronisation, phoneme-to-viseme mapping, facial
  landmark modelling and low-resource Bangla data make the problem highly
  specialised.
- **K4: Design and Development.** The project required systematic engineering
  design: end-to-end pipeline modelling, system architecture design and iterative
  development, with trade-offs between realism, latency and computation.
- **K5: Engineering Tools, Software and Techniques.** Modern tools were used —
  Python, PyTorch, computer vision and audio processing libraries — together with
  optimisation techniques to achieve real-time execution.
- **K6: Impact of Engineering on Society and Environment.** The project considers
  the ethical and social consequences of deepfakes, identity, privacy and
  responsible AI.
- **K7: Management, Finance and Project Management.** Planning, scheduling, version
  control and time management were used throughout, within limited resources.
- **K8: Lifelong Learning.** The field changes rapidly; new techniques, tools and
  research were studied and integrated during development.

P1 is satisfied, as K3, K4, K5, K6, K7 and K8 all apply.

**P2: Range of Conflicting Requirements.** The project involved many conflicting
requirements. A smaller model renders faster but risks poorer lip-sync; a larger
one looks better but cannot run on a laptop. A longer end-of-turn wait avoids
cutting the user off but makes the system feel slower. Emitting video frames
earlier reduces latency but risks lip frames ahead of their audio. Identity
preservation competes with generalisation to new speakers. These were balanced
through measurement and testing. P2 is satisfied.

**P3: Depth of Analysis Required.** The problem has no simple answer. Many methods
exist for audio-driven lip synchronisation — viseme-based, convolutional,
transformer and diffusion models — but none targets real-time Bangla lip-sync on
limited hardware. The solution required a comparative benchmark of published
systems, adaptation of the selected model, a causal diagnosis of why its mouth
moved without audio, and an analysis of why the standard metric could not detect
it. P3 is satisfied.

**P4: Familiarity of Issues.** Real-time Bangla lip synchronisation raises issues
not covered in standard coursework: audio–visual synchronisation under real-time
constraints, streaming audio features that must match offline extraction, Bangla
phoneme-to-viseme mapping, low-resource language data, identity-preserving
generation, and the reliability of evaluation metrics. P4 is satisfied.

**P5: Extent of Applicable Codes.** The project must follow professional and
ethical codes — the ACM and IEEE Codes of Ethics and responsible-AI principles —
with attention to data privacy, consent for facial data, and responsible use of
AI-generated media. P5 is satisfied.

**P6: Extent of Stakeholder Involvement.** Stakeholders include end users,
educators and other deploying institutions, developers, researchers and the wider
public, with differing needs for visual realism, real-time performance, ethical
use and reliability. P6 is satisfied.

**P7: Interdependence.** The audio processing, lip-motion model, rendering server,
conversational agent and real-time media layer are tightly interdependent and must
stay synchronised; latency, inference speed and ethical constraints follow directly
from the core problem. P7 is satisfied.

**Table 5.2: Mapping with complex problem solving**

| P1 Depth of Knowledge | P2 Conflicting Requirements | P3 Depth of Analysis | P4 Familiarity of Issues | P5 Applicable Codes | P6 Stakeholder Involvement | P7 Interdependence |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### 5.4.2 Engineering Activities

The engineering activities A1 to A5 were addressed through systematic analysis,
design, implementation and evaluation.

**A1: Range of Resources.** The solution combines many resources: speech input,
audio processing, a deep-learning lip-sync model, a rendering server, a
conversational model and a real-time media layer, all of which must be coordinated
to produce synchronised audio and video. Responsible data use, privacy and
deployment constraints are also in scope. A1 is satisfied.

**A2: Innovation.** The project applies audio-driven facial animation to Bangla
speech, which had not been done in the systems we reviewed. It identifies and
closes four performance gaps in a publicly available model, one of which made it
copy mouth shape instead of listening; introduces a Bangla phoneme-level evaluation; and shows,
with a silent-audio test and a real-video reference, that published systems leak
and that the standard metric cannot detect it. A2 is satisfied.

**A3: Level of Interaction.** Speech processing, deep-learning inference, rendering
and synchronisation interact closely, and balancing computation, visual quality and
responsiveness required continual adjustment across all of them. A3 is satisfied.

**A4: Consequences for Society and Environment.** Realistic talking avatars can
improve education, communication and access to services, but also raise concerns
about misuse and misinformation, which the project addresses through consent,
controlled use and transparency. Because the system runs on existing consumer
hardware, it avoids new hardware and limits its environmental impact. A4 is
satisfied.

**A5: Familiarity.** The problem requires knowledge beyond standard coursework:
deep-learning speech-driven synthesis, real-time streaming systems and evaluation
methodology. A5 is satisfied.

**Table 5.3: Mapping with complex engineering activities**

| A1 Range of Resources | A2 Innovation | A3 Level of Interaction | A4 Consequences for Society and Environment | A5 Familiarity |
|:-:|:-:|:-:|:-:|:-:|
| ✓ | ✓ | ✓ | ✓ | ✓ |

These activities show a structured and systematic approach to a complex
engineering problem, combining advanced AI techniques, real-time system design,
iterative optimisation and ethical awareness.

## 5.5 Summary

This chapter related the Real-Time Bangla Speech-Driven Avatar Generation system to
the relevant software, hardware, communication and societal standards, and
examined the economic, ethical and social constraints that shaped its design. The
cost analysis shows that the project can be delivered with very little expenditure.
Finally, the project was shown to be a complex engineering problem, because it
integrates real-time AI processing, low-resource language adaptation, neural
rendering on consumer hardware, and the evaluation of the models it depends on.

---

# Chapter 6: Conclusion

This chapter summarises the project, states its limitations, and describes the
journal paper in progress and future work.

## 6.1 Summary

This project set out to close a gap: there was no real-time conversational avatar
that speaks Bangla. It delivers one. A user speaks Bangla in a web browser, a
speech-to-speech agent replies in formal Bangla, and a 2D avatar speaks the reply
with synchronised lip movement.

We tested published lip-sync systems on Bangla video and selected SyncTalk_2D,
the only candidate small and fast enough for a laptop. We adapted it to Bangla
with Bangla training video, a Bangla-trained lip-sync expert and streaming audio
features. We then found four gaps in the model that limited its performance. The
most important was the mouth mask: the model copied the visible jaw instead of
listening, and its mouth moved in 41% of frames during silence. Our improved
model, **Alapon**,
moves in 0.1%, reconstructs the face more accurately, and separates open and
closed Bangla sounds almost as well as the real speaker.

We then checked whether published systems have the same problem. Five of the six
we tested leak on at least one dataset, and the standard lip-sync metric cannot
see it — it even scores three of them significantly above real video. These findings
are being prepared as a journal paper.

The completed system runs Alapon: 49 MB, 30 frames per second on a laptop RTX 3050
using 0.3 GB of GPU memory, and a reply within 1.5–2 seconds. On the Bangla test
clip its lip-sync score is 5.11 against 5.14 for real video, closer than any
published system tested.

**Table 6.1: Proposed against delivered**

| Proposed in FYDP-1 | Result |
|---|---|
| Real-time Bangla conversational avatar | **Done.** Working end-to-end system running Alapon; answers in 1.5–2 s. |
| Separate speech recognition, LLM and speech synthesis | **Changed** to one speech-to-speech model (Gemini 3.1 Flash Live), which is faster. |
| Real-time on local hardware | **Done.** 30 fps on a laptop RTX 3050, 0.3 GB GPU memory. |
| Bangla-trained SyncNet | **Done.** Retrained with mismatched pairs on the Bangla speaker and used during training. |
| Multilingual audio encoder (XLS-R) | **Tried, did not work.** Its lip-sync expert learned about half as well as with the default features; not used. |
| Bangla audiovisual dataset | **Collected.** 2.78 hours from 24 speakers; six speakers used as unseen test audio, none used for training. |
| Bangla phoneme-to-viseme mapping | **Defined and used for evaluation** (Table 3.3, Section 4.3.2); not yet used in training. |
| Lip-landmark loss | **Not done.** |
| *Not proposed:* model selection by benchmark | **Done.** Candidates compared on size and speed (Table 3.1). |
| *Not proposed:* closing the model's performance gaps | **Done.** Four gaps addressed; mouth movement during silence 41% → 0.1%. |
| *Not proposed:* leakage study of published systems | **Done.** Six checkpoints on Bangla and HDTF; journal paper in preparation. |

## 6.2 Limitation

- **One speaker.** Alapon is trained and tested on one Bangla speaker. Each new
  face needs its own video and training.
- **Training included the test frames.** The models in this report were trained
  before the training split was fixed, so Alapon's lip-sync, reconstruction and
  phoneme results on the test clip may be somewhat optimistic. The silence test
  and the six-speaker test are not affected. A retrain on the training frames only
  is in progress.
- **Mouth opening, not mouth shape.** The lips open and close correctly for Bangla
  sounds, but lip width did not show that rounded and spread sounds get different
  shapes.
- **Weaker on new voices.** For new speakers the open/closed contrast is 2.3×,
  against 4.0× for the real speaker on their own voice.
- **No human evaluation yet.** All results are automatic measurements; a study
  with native Bangla speakers is planned (Section 6.3.1).
- **Cloud dependency.** The avatar runs locally, but the conversation needs the
  Gemini API and an internet connection.
- **Corpus not used for training.** The 2.78-hour corpus was used only as test
  audio, and its source licences must be traced before it can be released.

## 6.3 Future Work

### 6.3.1 Journal Paper in Progress

Following our supervisor's advice, we are turning the findings of Sections 4.3.1
and 4.3.4 into a journal paper, **"Evaluation Integrity in Audio-Driven
Talking-Head Generation"**. Its argument is that a widely used talking-head model
was not driven by audio at all, that the field's standard evaluation ranked it as
working, and that the evaluation's failure modes are specific, measurable and
fixable. The target is *IEEE Transactions on Multimedia*, with *IEEE Transactions
on Circuits and Systems for Video Technology* and *Pattern Recognition* as
alternatives.

**Planned contributions**

1. **An analysis of SyncTalk_2D's performance gaps**: four gaps in the model's
   design and training, with a measured causal mechanism — a ten-pixel strip of jaw produced 91% of all mouth
   motion.
2. **Evidence that the standard lip-sync metric ranks generated video above real
   video** — three systems, statistically significant, on a public dataset, made
   visible by scoring real video as a reference row.
3. **A flaw in the leakage measure and its correction.** Leakage as currently
   measured depends on how much the test speaker moves their mouth; dividing by
   the speaker's own movement makes results comparable across speakers and
   datasets.
4. **A paired evaluation protocol** — leakage together with articulation — that
   no degenerate model can pass: a frozen mouth fails articulation, a copying
   model fails leakage, and an over-animated mouth fails articulation.
5. **A reproducible benchmark** — one scoring code for every system, frozen test
   splits, pinned environments, and a negative result reported in full.
6. **A Bangla phoneme-level evaluation**, and the Bangla corpus if its source
   licences can be traced.

**Status**

| Work | Status |
|---|---|
| Mask experiment with six models (Table 4.2) | Done |
| Benchmark: six published checkpoints on Bangla and HDTF, plus Alapon and the before-fix model on Bangla, one scoring code | Done |
| Silence test for every system on both datasets | Done |
| Bangla phoneme-level evaluation, including six unseen speakers | Done |
| Retrain Alapon and the before-fix model on the training frames only | In progress — about 15 GPU-hours |
| Human evaluation (below) | Planned — the critical path, 4–6 weeks |
| Further reconstruction metrics (LPIPS, FID) | Planned — about 3 days |
| Corpus licences and ethics approval | Planned |

**Human evaluation.** Native Bangla speakers will watch short (8-second) clips of
Alapon, the published systems and the real speaker, in random order and without
being told which is which, and rate each clip for how well the lips match the
speech and how natural it looks. A 6-person pilot will check the procedure, and
the main study will use at least 15 people. The ratings will then be compared
with the automatic measurements, to test whether the silence test and the
real-video reference agree with what people actually see better than the
standard lip-sync score does. If they do not, we will not propose them.

The remaining work is estimated at about six weeks, most of it the recruitment
for the human evaluation.

### 6.3.2 System Extensions

- **Bangla phoneme-level lip shape:** use the phoneme-to-viseme mapping as a
  training signal, so that different Bangla sounds produce visibly different
  mouth shapes, and narrow the gap on new voices (2.3× against 4.0×).
- **Multi-speaker training** with the collected Bangla corpus.
- **A Bangla audio encoder that works:** revisit XLS-R with more training data.
- **Fully offline operation** with an on-device conversational model.
- **Pilot deployment**, starting with Bangla-medium education.

---

# References

[1] Y. Zhou, X. Han, E. Shechtman, J. Echevarria, E. Kalogerakis, and D. Li,
"MakeItTalk: Speaker-aware talking-head animation," *ACM Transactions on
Graphics*, vol. 39, no. 6, Art. 221, 2020. doi: 10.1145/3414685.3417774

[2] Y. Guo, K. Chen, S. Liang, Y.-J. Liu, H. Bao, and J. Zhang, "AD-NeRF: Audio
driven neural radiance fields for talking head synthesis," in *Proceedings of the
IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021,
pp. 5784–5794.

[3] S. Yao, R. Zhong, Y. Yan, G. Zhai, and X. Yang, "DFA-NeRF: Personalized
talking head generation via disentangled face attributes neural rendering,"
arXiv preprint arXiv:2201.00791, 2022.

[4] Z. Peng, W. Hu, Y. Shi, X. Zhu, X. Zhang, H. Zhao, J. He, H. Liu, and Z. Fan,
"SyncTalk: The devil is in the synchronization for talking head synthesis," in
*Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR)*, 2024, pp. 666–676.

[5] K. Cho, J. Lee, H. Yoon, Y. Hong, J. Ko, S. Ahn, and S. Kim, "GaussianTalker:
Real-time talking head synthesis with 3D Gaussian splatting," in *Proceedings of
the 32nd ACM International Conference on Multimedia (ACM MM)*, 2024,
pp. 10985–10994. doi: 10.1145/3664647.3681627

[6] J. Li, J. Zhang, X. Bai, J. Zheng, X. Ning, J. Zhou, and L. Gu,
"TalkingGaussian: Structure-persistent 3D talking head synthesis via Gaussian
splatting," in *Computer Vision – ECCV 2024*, Lecture Notes in Computer Science,
Springer, 2024. doi: 10.1007/978-3-031-72684-2_8

[7] S. Aneja, A. Sevastopolsky, T. Kirschstein, J. Thies, A. Dai, and M. Nießner,
"GaussianSpeech: Audio-driven personalized 3D Gaussian avatars," in *Proceedings of
the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2025,
pp. 13065–13075.

[8] G. Kim, K. Seo, S. Cha, and J. Noh, "NeRFFaceSpeech: One-shot audio-driven 3D
talking head synthesis via generative prior," in *Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*, 2024.

[9] J. Li, J. Zhang, X. Bai, J. Zheng, J. Zhou, and L. Gu, "InsTaG: Learning
personalized 3D talking head from few-second video," in *Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2025,
pp. 10690–10700.

[10] K. R. Prajwal, R. Mukhopadhyay, V. P. Namboodiri, and C. V. Jawahar, "A lip
sync expert is all you need for speech to lip generation in the wild," in
*Proceedings of the 28th ACM International Conference on Multimedia (ACM MM)*,
2020, pp. 484–492. doi: 10.1145/3394171.3413532

[11] J. S. Chung and A. Zisserman, "Out of time: Automated lip sync in the wild,"
in *Computer Vision – ACCV 2016 Workshops*, Lecture Notes in Computer Science,
vol. 10117, Springer, 2017, pp. 251–263.

[12] Z. Zhang, L. Li, Y. Ding, and C. Fan, "Flow-guided one-shot talking face
generation with a high-resolution audio-visual dataset," in *Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2021,
pp. 3661–3670.

[13] A. Babu, C. Wang, A. Tjandra, K. Lakhotia, Q. Xu, N. Goyal, K. Singh,
P. von Platen, Y. Saraf, J. Pino, A. Baevski, A. Conneau, and M. Auli, "XLS-R:
Self-supervised cross-lingual speech representation learning at scale," in
*Proceedings of Interspeech*, 2022, pp. 2278–2282.
doi: 10.21437/Interspeech.2022-143

[14] C. Li et al., "LatentSync: Taming audio-conditioned latent diffusion models for lip
sync with SyncNet supervision," arXiv preprint arXiv:2412.09262, 2024. Code and
weights: https://github.com/bytedance/LatentSync

[15] Y. Zhang et al., "MuseTalk: Real-time high quality lip synchronization with latent space
inpainting," arXiv preprint arXiv:2410.10122, 2024. Code and weights:
https://github.com/TMElyralab/MuseTalk

[16] W. Zhong, C. Fang, Y. Cai, P. Wei, G. Zhao, L. Lin, and G. Li,
"Identity-preserving talking face generation with landmark and appearance
priors," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR)*, 2023, pp. 9729–9738.

[17] A. Bigata, R. Mira, S. Bounareli, M. Stypułkowski, K. Vougioukas,
S. Petridis, and M. Pantic, "KeySync: A robust approach for leakage-free lip
synchronization in high resolution," *Transactions on Machine Learning Research
(TMLR)*, 2025.

[18] Z. Peng et al., "SyncTalk_2D," GitHub repository.
https://github.com/ZiqiaoPeng/SyncTalk_2D

[19] Silero Team, "Silero VAD: Pre-trained enterprise-grade voice activity
detector," GitHub repository, 2021. https://github.com/snakers4/silero-vad

[20] LiveKit, "LiveKit Agents: A framework for building real-time voice AI
agents," GitHub repository. https://github.com/livekit/agents

[21] Google, "Gemini Live API," documentation.
https://ai.google.dev/gemini-api/docs/live

[22] arijitx, "wav2vec2-xls-r-300m-bengali: Bangla speech recognition model,"
Hugging Face model card.
https://huggingface.co/arijitx/wav2vec2-xls-r-300m-bengali

[23] W. Zhang, X. Cun, X. Wang et al., "SadTalker: Learning realistic 3D motion coefficients for stylized audio-driven
single image talking face animation," in *Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR)*, 2023, pp. 8652–8661.

[24] L. Tian, Q. Wang, B. Zhang, and L. Bo, "EMO: Emote portrait alive —
Generating expressive portrait videos with Audio2Video diffusion model under weak
conditions," in *Computer Vision – ECCV 2024*, Lecture Notes in Computer Science,
vol. 15141, Springer, 2024, pp. 244–260.

[25] A. Baevski, H. Zhou, A. Mohamed, and M. Auli, "wav2vec 2.0: A framework for
self-supervised learning of speech representations," in *Advances in Neural
Information Processing Systems (NeurIPS)*, vol. 33, 2020, pp. 12449–12460.
