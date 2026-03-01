# AEGIS Implementation Status + Fine-Tuning TODO

## What has been implemented from the spec

This repository now includes a runnable local Python implementation scaffold aligned with the architecture in `spec.md` (with fine-tuning deferred).

### 1) Project structure added

- `main.py` (root launcher)
- `aegis/main.py` (app bootstrap)
- `aegis/config.py` (`.env`-driven config)
- `aegis/ai/bedrock.py` (Bedrock reasoning client with tool-use loop)
- `aegis/ai/elevenlabs_speech.py` (ElevenLabs speech synthesis client)
- `aegis/graph/connection.py` (Neo4j async connection)
- `aegis/graph/primitives.py` (7 graph primitives from spec)
- `aegis/graph/tools.py` (tool handlers)
- `aegis/interface/text_ui.py` (text interface and possession status)
- `aegis/interface/voice_input.py` (Whisper input with typed fallback)
- `aegis/simulation/engine.py` (main loop + command handling)
- `aegis/simulation/possession.py` (cooldown + command duration logic)
- `aegis/simulation/spatial.py` (spatial helper)
- `aegis/prompts.py` (host response system prompt)

### 2) Environment variable setup (.env)

- `.env.example` created with all required keys.
- `.env` created for local runtime values.
- `.gitignore` includes `.env` and `.venv` so secrets and venv stay local.
- `AppConfig.from_env()` loads config from `.env` via `python-dotenv`.

### 3) Base Mistral model usage (no finetuning yet)

- Reasoning provider can be selected via `REASONING_PROVIDER`:
  - `bedrock` (default), model from `BEDROCK_REASONING_MODEL_ID`
  - `nvidia` (Serverless API), model from `NVIDIA_REASONING_MODEL_ID`
- Speech synthesis: ElevenLabs (`ELEVENLABS_API_KEY`, voice/model envs).
- Runtime writes generated speech audio clips to `artifacts/audio/`.

### 4) Virtual environment + dependencies

- `requirements.txt` added.
- `README.md` includes Windows PowerShell venv setup and run instructions.
- Code compile check passed (`python -m compileall .`).

## Fine-tuning work intentionally not implemented yet

The following is the remaining plan to add the fine-tuned speech model path from the spec.

### A) Data pipeline for SFT

1. Create `data/speech_sft/` dataset in chat format:
   - user: structured reasoning payload JSON
   - assistant: target `segments` JSON
2. Add validation script to enforce schema and max token length.
3. Add train/val split script and reproducible seed.

### B) Training scripts (LoRA/QLoRA)

1. Add training entrypoint (e.g. `scripts/train_speech_lora.py`) using:
   - `transformers`
   - `peft`
   - `trl` (`SFTTrainer`)
2. Base model target:
   - `mistralai/Mistral-7B-Instruct-v0.3`
3. Start with spec-aligned LoRA target modules:
   - `q_proj`, `k_proj`, `v_proj`, `o_proj`
4. Save adapter and tokenizer artifacts to `models/aegis-speech/`.

### C) Inference serving for finetuned model

1. Add vLLM launch command/docs to load merged/adapted model.
2. Version model id separately from base model (e.g. `aegis-speech-mistral-7b`).
3. Update runtime config to toggle between:
   - base model
   - finetuned model

### D) Evaluation + regression checks

1. Build eval set of command contexts and expected style/tone behavior.
2. Add script to compare base vs finetuned outputs:
   - JSON validity
   - tone adherence
   - latency
3. Add acceptance thresholds before enabling finetuned model by default.

### E) Operational guidance

1. Document minimum GPU VRAM and tested CUDA stack.
2. Add model artifact versioning + rollback instructions.
3. Keep API keys and credentials in `.env` only.

### F) NVIDIA Brev training plan (GPU)

1. Use Brev Console/CLI to create a GPU instance sized under budget.
2. Keep displayed hourly instance price at or below `$30/hr`.
3. Use checkpointed QLoRA runs and stop instances when idle.
4. Follow `BREV_FINETUNING_RUNBOOK.md` for step-by-step setup, run, and shutdown.

## Suggested next implementation phase

- Phase 1: Add dataset schema + validators.
- Phase 2: Add training script and smoke run.
- Phase 3: Add runtime model switch and evaluation harness.
