# Experiment results

## Runtime comparison (2026-04-05)

Hardware: Windows 11, NVIDIA RTX 3090 (24 GB VRAM)
Model: microsoft/VibeVoice-ASR-HF (8B params)
Commit: 8c5ab82 + uncommitted multi-runtime changes
Audio: 5 files (EN, RU), 10–20 seconds each, no prompt

### Summary table

| Metric | 4-bit (NF4) | 8-bit (INT8) | 16-bit (bf16) |
|---|---|---|---|
| Transcription quality EN | Correct | `[Unintelligible Speech]` | Correct |
| Transcription quality RU | Correct | Hallucination loop | Correct |
| Peak VRAM | 6,130 MB | 9,669 MB | 16,244 MB |
| Model load time | 29.2 s | 20.9 s | 12.2 s |
| Inference (EN, 19s audio) | 13.1 s | 11.6 s | 8.7 s |
| Inference (RU, 14s m4a) | 7.2 s | 56.4 s (stuck in loop) | 4.2 s |

### Observations

- **4-bit** is the practical default: good quality, fits comfortably in 24 GB with room for longer audio.
- **16-bit** produces identical transcriptions to 4-bit on these samples but is ~1.5–2x faster at inference. Uses 16.2 GB VRAM — tight on 24 GB for longer recordings, may need `--chunk-seconds`.
- **8-bit** is broken for this model. bitsandbytes emits `MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization` — the bf16→fp16 cast likely causes numerical issues. Output is either unintelligible or a hallucination loop. Not usable.

### Conclusion

Use **4-bit** as the default runtime. Use **16-bit** when VRAM allows and speed matters. **8-bit is not viable** for VibeVoice-ASR with current bitsandbytes.
