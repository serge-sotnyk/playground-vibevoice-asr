# Experiment results

## Experiment 1: Runtime comparison (2026-04-05)

Hardware: Windows 11, NVIDIA RTX 3090 (24 GB VRAM)
Model: microsoft/VibeVoice-ASR-HF (8B params)
Audio: 5 files (EN, RU), 10–20 seconds each, no prompt

### Summary table

| Metric | 4-bit (NF4) | 8-bit TorchAO | 8-bit Quanto | 16-bit (bf16) |
|---|---|---|---|---|
| Transcription quality EN | Correct | Correct | Correct | Correct |
| Transcription quality RU | Correct | Correct | Correct | Correct |
| Peak VRAM | 6,130 MB | 9,438 MB | 9,424 MB | 16,244 MB |
| Model load time | 29.2 s | 12.7 s | 12.7 s | 12.2 s |
| Inference (EN, 19s mp3) | 13.1 s | 10.6 s | 11.3 s | 8.7 s |
| Inference (RU, 14s m4a) | 7.2 s | 5.5 s | 6.0 s | 4.2 s |
| Inference (RU, 65s mp3) | — | 29.4 s | 33.3 s | — |
| CLI flag | `-r 4` | `-r 8` | `-r 8q` | `-r 16` |

### Observations

- **4-bit** is the practical default: good quality, lowest VRAM (6.1 GB), fits comfortably in 24 GB with room for longer audio. Slowest model load (29s) due to bitsandbytes quantization overhead.
- **16-bit** produces identical transcriptions to 4-bit on these samples but is ~1.5–2x faster at inference. Uses 16.2 GB VRAM — tight on 24 GB for longer recordings, may need `--chunk-seconds`.
- **8-bit TorchAO** (Int8WeightOnly): correct transcription, ~9.4 GB VRAM. Good balance between 4-bit VRAM savings and 16-bit speed. Faster load than 4-bit.
- **8-bit Quanto** (INT8 weights): nearly identical to TorchAO in quality and VRAM. Slightly slower inference. Better macOS compatibility.

### Failed approaches

- **bitsandbytes INT8** (`BitsAndBytesConfig(load_in_8bit=True)`): broken for this model. Emits `MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16` — the bf16→fp16 cast causes numerical issues. Output is either `[Unintelligible Speech]` or a hallucination loop. Replaced with TorchAO and Quanto.

### Conclusion

Use **4-bit** as the default runtime (lowest VRAM). Use **16-bit** when VRAM allows and speed matters. For 8-bit, use **TorchAO** (`-r 8`) as the faster option or **Quanto** (`-r 8q`) for macOS.
