# Documentation

Two tracks, kept apart because they answer different questions and are judged by
different people.

| | Asks | Audience |
|---|---|---|
| **[fydp/](fydp/)** | Does the system work, and why is it useful? | FYDP judges, supervisor |
| **[research/](research/)** | What does it show about how the field measures itself? | Journal reviewers |

---

## FYDP — the system

| Doc | What it is |
|---|---|
| [report/latex/](fydp/report/latex/) | **Final report** in the official UIU LaTeX format (Overleaf zip beside it); [report/README](fydp/report/README.md) lists what changed, the remaining TODOs, and the source of every number |
| [benchmark_result.md](fydp/benchmark_result.md) | **Every result table in one place** — Bangla comparison, mask fix, speed, Bangla phonemes, HDTF |
| [story.md](fydp/story.md) | Defence pitch: the gap, the numbers, the use cases, objections, and the pre-show verification checklist |

**How to run it** lives in the root [README](../README.md), not here — that's the
repository's front page and the first thing anyone cloning it reads.

## Research — the paper

| Doc | What it is |
|---|---|
| [proposal.md](research/proposal.md) | Journal proposal (IEEE TMM): thesis, results on both datasets, the falsified hypothesis, contributions, risks |
| [roadmap.md](research/roadmap.md) | Execution stages, what each found, open decision points, carried-forward traps |
| [human_study_plan.md](research/human_study_plan.md) | 6-person pilot, then ≥15 native Bangla speakers |

## Docs that live beside their code

Kept next to what they describe, so they move with it:

| Doc | Covers |
|---|---|
| [benchmark/README.md](../benchmark/README.md) | The cross-system scorer: validation table, alignment offsets, known traps |
| [hdtf/README.md](../hdtf/README.md) | The HDTF test subset: selection rule, attrition, the defective clip |
| [SyncTalk_2D/README.md](../SyncTalk_2D/README.md) | The upstream model that Alapon is built on |

## Not in the repository

`research/` at the repository root is a **local-only** workspace — corpus, frozen
environments, logs, and planning notes — and is gitignored on purpose. Its planning
docs (`PLAN.md`, `CHECKLIST.md`, `CONTEXT.md`, `PUBLICATION.md`) date from
19–21 August and predate the benchmark results; where they disagree with
`docs/research/proposal.md`, the `docs/` version is current.
