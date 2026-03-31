# VibeVoice-ASR local experiment plan for recruiter-harness

## 1. What this repo is for

Create a separate experimental repository whose only purpose is to evaluate whether **VibeVoice-ASR** can become the local transcription + diarization backend for the recruiter-harness workflow.

This repo is **not** the production integration yet.
It is a focused playground for:

- loading and running VibeVoice-ASR locally on a Windows machine with RTX 3090
- testing **bitsandbytes 4-bit first**
- falling back to the official Hugging Face Transformers path if 4-bit is unstable or unsupported
- evaluating quality on real recruiter / interview audio
- checking whether prompt-based glossary / hotword hints improve results enough
- deciding whether the final result is worth integrating into the main private harness repository

The repo may remain a standalone example repo. Packaging is **optional** and should be deferred until the experiment proves useful.

---

## 2. High-level decisions already made

### Chosen primary direction
Use **VibeVoice-ASR** as the first model family to evaluate.

### Runtime priority
1. **First attempt** – Hugging Face / Transformers + **bitsandbytes 4-bit**
2. **Fallback** – Hugging Face / Transformers in normal precision
3. **Do not start with vLLM**

vLLM is explicitly out of scope for the initial pass unless the first two paths succeed and there is a strong reason to expose a local API service later.

### Packaging decision
Do **not** force packaging from the start.
Keep the repo simple.
At the end, decide whether to:

- keep it as an example repo
- extract a reusable internal module
- convert it into a package

### Integration target
The final consumer is a private recruiter-harness repository that turns raw notes and transcripts into normalized markdown for Obsidian.
This experiment only needs to produce a clean and reliable **structured transcript output** that can later be consumed by that harness.

---

## 3. What success looks like

The experiment is successful if we can process typical recruiter / interview audio locally and obtain:

- usable transcription quality
- acceptable diarization quality for 2-speaker conversations
- stable execution on RTX 3090
- a practical way to inject glossary / term hints
- structured output that is easy to post-process into harness markdown

“Usable” means the result is good enough for downstream summarization and structured note generation, even if it is not perfect word-for-word.

---

## 4. Non-goals

These are explicitly **not** required in the first version:

- production-ready packaging
- public PyPI release
- vLLM deployment
- web UI
- automatic file watchers
- background services
- generalized support for many ASR backends
- full benchmark suite against all possible models

This repo should stay focused.

---

## 5. Constraints and assumptions

### Hardware
- Windows workstation
- NVIDIA RTX 3090
- local execution only

### Audio reality
Typical input is:

- recruiter calls
- interviews
- meetings recorded on dictaphone / phone / desktop tools
- mostly one dominant language per recording
- languages often include **English, Ukrainian, Russian**
- English terms and product names may appear inside Ukrainian/Russian speech
- diarization is important
- quick speaker switching may happen

### Expected transcript shape
The output should preserve:

- speaker segmentation
- timestamps
- text content
- enough structure to identify who spoke when

Speaker labels may remain generic like `speaker_0` / `speaker_1` in the first version.
Mapping them to actual people can be a later post-processing step.

---

## 6. Main deliverables for this experiment repo

At the end of the work, the repo should contain a small but complete reference implementation with:

1. **A runnable CLI entry point** for local transcription
2. **Configuration support** for model mode and glossary prompt
3. **One canonical structured output format**
4. **A small evaluation workflow** on a few real or representative audio samples
5. **A short README** describing how to run the experiments and interpret results
6. **A decision note** saying whether to integrate into the private harness repo

Keep the codebase small and easy to inspect.

---

## 7. Recommended repo structure

The exact file layout is flexible, but the repo should end up conceptually containing these parts:

- a small CLI layer
- a thin transcription engine layer
- config loading
- prompt / glossary preparation
- output serialization
- evaluation helpers
- sample configs and example commands

Avoid overengineering.
Do not build abstractions for hypothetical future engines unless they are needed for the actual fallback path.

---

## 8. The output format to standardize on

Pick **one** normalized JSON format for the repo and use it everywhere.

Suggested logical schema:

- source audio path
- model identifier
- runtime mode
- language hints if any
- prompt / glossary used
- full transcript text
- segment list
  - speaker
  - start time
  - end time
  - text
- optional metadata
  - processing duration
  - device info
  - warnings
  - errors

The harness integration can later transform this JSON into markdown for Obsidian.
The experiment repo should not tightly couple itself to the harness markdown schema.

---

## 9. Glossary / prompt strategy

Support a small glossary file from the start.

The glossary should not be a giant raw dictionary.
Instead split it into categories such as:

- person names
- company names
- products / tools
- acronyms
- domain-specific terms

Build a helper that turns this glossary into a short prompt/context string suitable for VibeVoice-ASR.

Important:

- keep it short
- keep it deterministic
- prefer the most relevant terms per recording instead of dumping everything

The first version does not need smart retrieval.
A simple config-driven glossary is enough.

---

## 10. Phased execution plan

## Phase 0 – repo bootstrap

Use the existing bootstrap conventions already known by the initialization agent.
No need to redesign those instructions here.
Only add the experiment-specific requirements.

Expected outcome:

- clean experimental repo created
- Python environment managed with uv
- reproducible run commands
- GPU dependencies documented

Stop here only if the environment cannot be made reproducible.

---

## Phase 1 – baseline implementation with official HF path

Even though bitsandbytes is the preferred first runtime, implement the code structure so that the **official Transformers path** exists as the fallback baseline.

Goal of this phase:

- prove that the repo can run VibeVoice-ASR locally end-to-end
- load a sample audio file
- pass optional prompt/glossary context
- return parsed structured output
- save normalized JSON result

This phase should establish the reference implementation and expected output schema.

Expected outcome:

- one successful transcription on a real sample
- one saved JSON result
- clear CLI command that works

This phase is complete once the non-quantized fallback path is known to work.

---

## Phase 2 – bitsandbytes 4-bit path

Add the preferred runtime path using **bitsandbytes 4-bit**.

Goal of this phase:

- try loading the HF-compatible VibeVoice-ASR checkpoint using bitsandbytes quantization
- verify that inference runs on the target machine
- compare it against the fallback baseline

Measure at least:

- whether model loading succeeds
- whether inference completes without crashes or severe corruption
- peak VRAM usage if practical
- wall-clock processing time
- output quality relative to the baseline on the same files

Expected outcome:

- either a working 4-bit mode
- or a documented failure / instability that justifies falling back to normal precision

Important:

- do not assume 4-bit is automatically faster
- memory reduction is useful even if speed is similar
- if 4-bit quality is visibly worse, document that honestly

This is the key decision phase.

---

## Phase 3 – prompt / glossary experiments

Once at least one runtime path works, test whether prompt-based hints help enough to matter.

Create a small set of controlled experiments:

1. no prompt
2. minimal prompt with speaker context
3. prompt with recruiter + candidate names
4. prompt with names + company/tool glossary

Evaluate:

- spelling of names
- spelling of technical products
- handling of English terms inside Ukrainian/Russian speech
- whether prompt improves or harms recognition

Expected outcome:

- a practical recommendation for default prompt format
- evidence about whether glossary support is worth keeping

---

## Phase 4 – evaluation on representative audio

Prepare a small curated audio set.
Do not overdo it.
A compact set is enough if it is representative.

Recommended categories:

- clean 2-speaker recruiter call in English
- Ukrainian or Russian conversation with English technical terms
- messier recording with weaker audio quality
- one recording with faster turn-taking

For each sample, capture:

- runtime mode used
- approximate duration of audio
- subjective transcript quality
- subjective diarization quality
- obvious recurring failure modes

This does not need a formal academic benchmark.
It needs to be enough to support a practical engineering decision.

---

## Phase 5 – integration readiness decision

After the above phases, write a concise decision note.

Possible outcomes:

### Outcome A – good enough to integrate
Choose this if:

- at least one runtime path is stable
- quality is acceptable for downstream processing
- diarization is useful enough in practice
- the JSON output is easy to consume

Then define the minimal integration contract for the private recruiter-harness repo.

### Outcome B – promising but not ready
Choose this if:

- some parts work
- quality is inconsistent
- or setup is too fragile

Then keep the repo as a research spike and document the blockers.

### Outcome C – reject VibeVoice-ASR for now
Choose this if:

- stability is poor
- diarization is not reliable enough
- multilingual switching is too problematic
- or setup complexity outweighs the value

If this happens, preserve the repo as a useful reference and stop.

---

## 11. Experiment matrix

Use a small matrix, not a giant one.

Minimum matrix:

### Runtime modes
- HF official, normal precision
- HF + bitsandbytes 4-bit

### Prompt modes
- none
- short contextual prompt
- short contextual prompt + glossary terms

### Audio categories
- clean
- mixed-language with anglicisms
- noisy / harder sample

That is enough to make a decision.

---

## 12. What to record for each run

Every experimental run should capture, at minimum:

- timestamp
- git commit or code version
- model name
- runtime mode
- prompt mode
- input audio identifier
- processing duration
- success / failure
- output file path
- notes about quality

A simple markdown log or JSONL file is enough.
Do not build a database for this.

---

## 13. Acceptance criteria

The repo is ready for handoff when all of the following are true:

1. A new machine checkout can be set up reproducibly from the repo instructions
2. At least one runtime path works end-to-end on target hardware
3. The CLI can transcribe a file and save normalized JSON
4. Prompt / glossary support is implemented and tested at least minimally
5. A small evaluation set has been run
6. There is a written decision note about whether to integrate

If any of these are missing, the repo is not done.

---

## 14. Reasonable stop points

It is acceptable to stop at the following points:

### Stop point 1 – technical feasibility proven
Stop after:

- one runtime path works
- one real audio file is transcribed successfully
- normalized JSON output exists

This is enough to prove the idea is viable.

### Stop point 2 – engineering decision ready
Stop after:

- baseline + bitsandbytes compared
- prompt / glossary tested
- representative audio tested
- written conclusion exists

This is the preferred stop point.

### Stop point 3 – ready for private integration
Stop after:

- minimal contract for downstream harness integration is documented
- the private repo can consume the JSON output with little extra work

Anything beyond this is optional.

---

## 15. Recommended implementation style

Keep the implementation boring and explicit.

Preferred characteristics:

- small modules
- minimal indirection
- config over hardcoding where practical
- no framework-heavy architecture
- easy CLI usage
- clear logs and errors

Avoid building a large abstraction layer for future ASR engines until that need is real.

---

## 16. Suggested CLI behavior

The repo should expose one simple command that accepts at least:

- input audio path
- output path
- runtime mode
- optional prompt file or prompt text
- optional glossary file
- optional experiment label

Example intent, not exact syntax:

- transcribe one audio file
- choose quantized or fallback mode
- save normalized JSON
- optionally save a human-readable markdown preview

Markdown preview is optional. JSON is mandatory.

---

## 17. Error handling expectations

The tool should fail clearly when:

- model loading fails
- CUDA is unavailable
- audio file is unreadable
- output cannot be serialized
- prompt or glossary file is invalid

Prefer explicit and useful errors over silent fallback behavior.

---

## 18. Notes about diarization expectations

Diarization should be treated as useful but imperfect.
The evaluation should focus on whether diarization is good enough for recruiter-call notes, not whether it is perfect under heavy overlap.

The first version does not need:

- speaker identity resolution
- voiceprint matching
- persistent speaker profiles

Only segment-level diarization quality matters initially.

---

## 19. Notes about multilingual behavior

Because quick within-speaker language switching may be imperfect, evaluate it explicitly but pragmatically.

Questions to answer:

- is the result still understandable enough for summarization?
- are product names and English terms preserved reasonably?
- does the glossary help enough to justify using it?

Do not block the whole project on perfect code-switching if the overall output is still operationally useful.

---

## 20. Final handoff artifact expected from this repo

At the end of the experiment, produce one concise handoff note containing:

- which runtime mode won
- what hardware it was tested on
- what kinds of audio were tested
- known failure modes
- whether glossary prompting helped
- whether the result is good enough for recruiter-harness integration
- what the private harness will need from this repo

That handoff note is the main deliverable for the next integration agent.

---

## 21. Explicit implementation order

Follow this order unless a blocker forces a change:

1. Bootstrap repo
2. Define normalized output schema
3. Implement official HF baseline path
4. Verify one successful real transcription
5. Add bitsandbytes 4-bit mode
6. Compare baseline vs 4-bit
7. Add prompt / glossary support
8. Run representative sample set
9. Write decision note
10. Only then decide whether packaging is worth it

---

## 22. Final instruction to the implementation agent

Do not try to make this perfect or overly generic.
The goal is to determine whether VibeVoice-ASR is a practical local transcription backend for recruiter-related audio in a multilingual real-world workflow.

A small, honest, testable implementation is better than a polished but speculative architecture.

## Links
* HF model card VibeVoice-ASR-HF — https://huggingface.co/microsoft/VibeVoice-ASR-HF
* Microsoft VibeVoice repo — https://github.com/microsoft/VibeVoice
* Microsoft vLLM notes — https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-vllm-asr.md
* Transformers bitsandbytes docs — https://huggingface.co/docs/transformers/quantization/bitsandbytes
* bitsandbytes installation — https://huggingface.co/docs/bitsandbytes/main/installation
