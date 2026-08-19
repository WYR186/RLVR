# Blocker — artifacts can leave a Colab runtime but cannot get back in

**Date:** 2026-08-18
**Status:** This is the binding constraint on finishing the MVP. It is not a bug
to fix in code; it needs one action only the operator can take.

---

## 1. What happened

The runtime disconnected at ~18:20 with:

> "Your runtime has been disconnected due to inactivity **or reaching its
> maximum duration**."

It had been alive since ~08:45 — about **9.5 hours** — and was polled every 15
minutes throughout, so this was almost certainly the **maximum-duration** limit,
not idle reclamation. That distinction matters: the 15-minute keep-alive works,
but nothing defeats a hard session cap.

Lost with it: the in-flight Stage-B length measurement and the partial
`ckpt0_seed42` artifact files. **Nothing else** — Stage A's checkpoints and all
of Phase 2 were already downloaded to the operator's machine.

## 2. The structural problem this exposes

| direction | mechanism | works? |
|---|---|---|
| runtime -> Mac | `google.colab.files.download()` | yes (used for ~485 MB today) |
| Mac -> runtime | Google Drive mount | **blocked** — OAuth popup blocked by Chrome |
| Mac -> runtime | `file_upload` browser tool | **impossible** — the tool caps a call at **10 MB** and only accepts files shared with the session; each adapter is 154–165 MB |
| Mac -> runtime | git | **no** — `*.safetensors` is gitignored, and 165 MB exceeds GitHub's per-file limit without LFS |

So Stage-A checkpoints are safe on disk but **cannot be put back into a Colab
runtime**.

## 3. Why that stops Delta-R specifically

| Stage-B arm | needs | reproducible without upload? |
|---|---|---|
| ckpt-0 | base Instruct + a fresh LoRA adapter | **yes** — this is exactly what `run_stage_b_adaptation(stage_a_checkpoint=None)` builds |
| ckpt-50 | the trained ckpt-50 adapter | **no** |
| ckpt-100 | the trained ckpt-100 adapter | **no** |

Delta-R is `R_B(ckpt_t) − R_B(ckpt_0)`. Two of its three points are unreachable.

**And "just re-run Stage A first, in the same session" does not work:** Stage A
is 5.5 h and Stage B is ~7.5 h, so one session would need ~13 h against an
observed ceiling of ~9.5 h. The two phases *cannot* share a runtime. Persistence
between sessions is therefore mandatory, not a convenience.

## 4. The one action that unblocks it

**Mount Google Drive once.** It fixes both directions permanently and for every
future session: Stage-A checkpoints get written there during training, and any
later runtime reads them straight back.

The obstacle is only that Chrome blocks Colab's OAuth popup. Allowing pop-ups
for `colab.research.google.com` and completing the Google consent screen is a
~30-second action, but it must be done by the operator — the consent screen
carries session credentials and the auth URL is deliberately withheld from the
assistant.

### Alternative if Drive stays unavailable

Push the three adapters to a **private Hugging Face Hub repo** and pull them
with `snapshot_download` in each new runtime. This needs an HF write token
placed in Colab Secrets by the operator. It is strictly worse than Drive — one
more credential, and the GitHub-PAT path already failed this project the same
way — but it does not depend on the popup.

## 5. What is NOT being done, and why

No Stage-A re-run has been started. Re-running 5.5 h of A100 would produce
checkpoints in a runtime that will hit its duration cap before Stage B finishes,
and they would then be as stranded as the current ones. **Spending compute
before the persistence problem is solved just reproduces the problem at a higher
cost.**
