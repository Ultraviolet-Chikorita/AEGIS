# NVIDIA Brev Fine-Tuning Runbook (AEGIS)

This runbook is for fine-tuning the AEGIS speech model on NVIDIA Brev while keeping spend below your **$30/hour** budget target.

## Source docs

- Brev docs root: https://docs.nvidia.com/brev/latest/
- Brev Quickstart: https://docs.nvidia.com/brev/latest/quick-start.html
- Brev Console docs: https://docs.nvidia.com/brev/latest/console.html
- Brev CLI docs: https://docs.nvidia.com/brev/latest/brev-cli.html

## 1) Provision a Brev GPU instance

From Brev Quickstart, base flow is:

1. Create account in Brev console (`https://brev.nvidia.com`).
2. Create New Instance.
3. Select compute.
4. Configure + Deploy.
5. Connect to instance.

CLI access patterns (from docs):

```bash
brev shell <instance-name>
brev open <instance-name>
```

## 2) Pick compute under the $30/hr cap

When selecting the GPU in Brev:

1. Keep **displayed hourly price <= $30/hr**.
2. Prefer smallest GPU that can run QLoRA for Mistral 7B.
3. Avoid overprovisioning CPU/RAM/storage unless required by data size.

Practical guidance:

- Start with a single-GPU setup.
- Use QLoRA (4-bit) before considering full fine-tune.
- Scale up only if OOM or throughput is unacceptable.

## 3) Prepare the instance

Inside the Brev instance:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install torch transformers datasets peft trl bitsandbytes accelerate
```

> Note: keep fine-tuning libs in the training environment only; app runtime does not need all of them.

## 4) Move data/code to Brev

Use git clone/pull or SCP (see Brev SCP docs).

Recommended structure:

- `data/speech_sft/train.jsonl`
- `data/speech_sft/val.jsonl`
- `scripts/train_speech_lora.py`
- `models/aegis-speech/` (outputs)

## 5) Run training with checkpointing

Use frequent checkpoints so interrupted runs can resume.

Example command pattern:

```bash
python scripts/train_speech_lora.py \
  --train_file data/speech_sft/train.jsonl \
  --val_file data/speech_sft/val.jsonl \
  --output_dir models/aegis-speech \
  --save_steps 200 \
  --eval_steps 200 \
  --logging_steps 20
```

## 6) Budget guardrails (must-do)

1. **Set a timer** before each run block (e.g. 45-90 min blocks).
2. **Stop instance immediately** when not actively training/evaluating.
3. Record start/end times and effective $/run in a small log file.
4. Resume from latest checkpoint instead of restarting from scratch.

Suggested simple run log (`training_runs.md`):

- instance type
- hourly rate shown in Brev
- start time / stop time
- elapsed hours
- estimated cost
- checkpoint used
- val metrics snapshot

## 7) End-of-session shutdown checklist

Before leaving the session:

1. Sync checkpoints/artifacts back to persistent storage.
2. Verify latest checkpoint can be resumed.
3. Stop (or delete) Brev instance.
4. Save run cost + metrics note.

## 8) Integration after training

Once fine-tuned model quality is accepted:

1. Export/merge adapter as needed.
2. Serve model behind your local inference endpoint (or chosen serving stack).
3. Add runtime switch to choose `base` vs `finetuned` speech model.
4. Keep `ELEVENLABS` path available as fallback while validating finetuned output.
