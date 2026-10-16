# MA-Bench

Reference code for MA-Bench, a benchmark of how well language models perform mission analysis, the first step of
military operational planning. The benchmark itself (ten synthetic scenario packets, 1,257 expert-written rubric
items, the frozen judge prompt and scorer, and every graded brief and verdict) is on Hugging Face at
[ScaleAI/MA-Bench](https://huggingface.co/datasets/ScaleAI/MA-Bench). This repository runs the protocol on new
models and reproduces the published numbers from the released verdicts.

| module | what it does |
|---|---|
| `ma_bench.generate` | one brief per scenario, model and seed over any OpenAI-compatible endpoint, with the 3,000-word cap |
| `ma_bench.grade` | grades briefs with the frozen judge prompt, one call per brief, failures-only structured output |
| `ma_bench.score` | weighted and unweighted scores per condition and model, with 95% bootstrap intervals over scenarios |
| `ma_bench.agreement` | the judge against the four-expert panel, and judges against each other |

## Setup

```bash
pip install -r requirements.txt huggingface_hub   # or: pip install -e . huggingface_hub
hf download ScaleAI/MA-Bench --repo-type dataset --local-dir MA-Bench
```

The `hf` command ships with `huggingface_hub` 1.0 and later.

Every command takes `--release MA-Bench`, the local copy of the dataset. Generation and grading check the release's
`data/` and `scoring/` files against `MANIFEST.json` before the first call.

## Reproduce the published numbers

No API calls are needed; these read `runs/verdicts.jsonl` and `runs/expert_study.jsonl` from the release.

```bash
python -m ma_bench.score --release MA-Bench --judge gpt-5.6          # main results, all conditions
python -m ma_bench.agreement experts --release MA-Bench              # judge vs the expert panel
python -m ma_bench.agreement judges --release MA-Bench               # three-judge agreement table
```

## Run a new model

Set `OPENAI_BASE_URL` and `OPENAI_API_KEY` for an OpenAI-compatible endpoint (a provider's API, or a
[LiteLLM](https://github.com/BerriAI/litellm) proxy for several providers), then:

```bash
python -m ma_bench.generate --release MA-Bench --models <model-id> --seeds 3 \
  --reasoning-effort medium --other-reasoning-effort medium --out runs/my_model
python -m ma_bench.grade --release MA-Bench --briefs runs/my_model/briefs.jsonl --out runs/my_model \
  --judge openai/gpt-5.6
python -m ma_bench.score --release MA-Bench --verdicts runs/my_model/verdicts_openai_gpt_5_6.jsonl \
  --briefs runs/my_model/briefs.jsonl
```

Both steps cache every call under `--out`, so an interrupted run resumes. The outputs use the release's
`runs/briefs.jsonl` and `runs/verdicts.jsonl` layouts, so a new model can be scored next to the published ones.

## Protocol

- **Input.** One user message: the task prompt with the operation name filled in, the length-cap paragraph, a blank
  line, `# SOURCE DOCUMENTS`, then the packet assembled from `data/documents.jsonl` in `doc_order`
  (`ma_bench.release.Release.model_input`). The assembly is byte-identical to the input used for the published runs.
- **Length cap.** The model is told the 3,000-word limit, and text beyond the 3,000th word is cut before grading. A word
  is a whitespace-delimited token containing a letter or digit. `--word-cap 0` runs the uncapped ablation, with no cap
  paragraph and no cut; label such runs `uncapped`.
- **Settings.** Medium reasoning effort where the provider exposes it, otherwise the provider default. No temperature,
  since several models reject it and provider defaults are the reproducible choice. Seeds differ through ordinary
  sampling.
- **Judge.** `openai/gpt-5.6` at high reasoning effort with the frozen prompt in `scoring/judge_prompt.md`. The judge sees
  the task, the case header and the scenario's full rubric, never the packet, and lists only the items a brief fails.
  The judge input is built by the release's own `scoring/judge_input.py`.
- **Score.** Severity weights CI +9, I +5, SI +2, D -5, CD -9. A negative item names a mistake and counts against the
  brief when the mistake is present. Weighted score = (positive weight earned + negative weight incurred) / total
  positive weight, not clamped. A model's score is the mean over scenarios of its mean brief score.

## License

The code is released under the MIT License (see `LICENSE`). The dataset carries its own license on Hugging Face.
