"""Generate MA-Bench briefs: one chat completion per (scenario, model, seed) over any OpenAI-compatible endpoint.

The model gets the task prompt, the length-cap paragraph and the full packet in a single user message, and its reply
is the brief. In capped runs the brief is cut after the 3,000th word before it is saved, as the protocol requires.
Every completion is cached under <out>/cache, so an interrupted run resumes where it stopped.

Environment: OPENAI_BASE_URL and OPENAI_API_KEY for the endpoint (a provider's API or a LiteLLM proxy).

Example:
  python -m ma_bench.generate --release MA-Bench --models anthropic/claude-opus-5 openai/gpt-5.6 --seeds 3 \
    --reasoning-effort medium --other-reasoning-effort medium --out runs/my_run
Writes <out>/briefs.jsonl in the release's runs/briefs.jsonl layout.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

from ma_bench.release import PROTOCOL_WORD_CAP, Release, cap_words, slug, write_jsonl

MAX_OUTPUT_TOKENS = 49152  # reasoning + visible output; some providers count hidden thinking against max_tokens
TIMEOUT_S = 600.0
MIN_BRIEF_CHARS = 500  # shorter replies are treated as failed requests and retried, never cached


def is_reasoning_model(model: str) -> bool:
  m = model.lower()
  return any(k in m for k in ("gpt-5", "gpt-6", "o1", "o3", "o4", "reasoning", "deepseek-r"))


def request_kwargs(model: str, a) -> dict:
  """No temperature: several models reject it, and provider defaults are the reproducible choice."""
  if is_reasoning_model(model):
    kw = {"max_completion_tokens": MAX_OUTPUT_TOKENS}
    effort = a.reasoning_effort
  else:
    kw = {"max_tokens": MAX_OUTPUT_TOKENS}
    effort = a.other_reasoning_effort
  if effort:
    kw["reasoning_effort"] = effort
  if a.extra_body:
    kw["extra_body"] = json.loads(a.extra_body)
  return kw


def is_rate_limit(exc: Exception) -> bool:
  s = f"{type(exc).__name__} {exc}".lower()
  return "ratelimit" in s or "429" in s or "rate limit" in s


async def with_retries(factory, *, label: str, tries: int = 6, base: float = 20.0, rl_tries: int = 40, rl_wait: float = 60.0):
  """Retry transient errors with exponential backoff; rate limits get a separate, longer budget."""
  n_err = n_rl = 0
  while True:
    try:
      return await factory()
    except Exception as exc:  # noqa: BLE001
      if is_rate_limit(exc):
        n_rl += 1
        if n_rl > rl_tries:
          raise
        wait = rl_wait * min(4, n_rl) + random.uniform(0, 15)
      else:
        n_err += 1
        if n_err >= tries:
          raise
        wait = base * (2 ** (n_err - 1))
        print(f"  retry {n_err}/{tries - 1} {label}: {type(exc).__name__}: {str(exc)[:120]} (sleep {wait:.0f}s)", flush=True)
      await asyncio.sleep(wait)


async def generate_one(client, sem, R: Release, case_id: str, model: str, seed: int, a, cache: Path) -> dict:
  f = cache / f"gen__{case_id}__{slug(model)}__s{seed}.json"
  if f.exists():
    return json.loads(f.read_text(encoding="utf-8"))
  messages = [{"role": "user", "content": R.model_input(case_id, a.word_cap)}]
  kwargs = request_kwargs(model, a)

  async def request():
    resp = await client.chat.completions.create(model=model, messages=messages, **kwargs)
    content = resp.choices[0].message.content or ""
    if len(content.strip()) < MIN_BRIEF_CHARS:
      raise RuntimeError(f"short reply ({len(content)} chars), finish_reason={resp.choices[0].finish_reason}")
    return resp

  async with sem:
    t0 = time.time()
    resp = await with_retries(request, label=f"{case_id} {model} s{seed}")
  raw = resp.choices[0].message.content or ""
  text, raw_words, capped = cap_words(raw, a.word_cap)
  rec = {
    "brief_id": f"{a.condition}/{R.scenario_code(case_id)}__{slug(model)}__s{seed}",
    "condition": a.condition,
    "case_id": case_id,
    "scenario": R.scenario_code(case_id),
    "model": a.names.get(model, model),
    "model_slug": slug(model),
    "seed": seed,
    "text": text,
    "raw_text": raw,
    "raw_words": raw_words,
    "capped": capped,
    "word_cap": a.word_cap,
    "served_model": getattr(resp, "model", None),
    "finish_reason": resp.choices[0].finish_reason,
    "usage": resp.usage.model_dump() if getattr(resp, "usage", None) else None,
    "request": kwargs,
    "secs": round(time.time() - t0, 1),
  }
  tmp = f.with_suffix(".json.tmp")
  tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
  tmp.replace(f)
  print(f"  {case_id:4s} {model:36s} s{seed} {raw_words:>6,} words{' (capped)' if capped else ''} {rec['secs']:6.0f}s", flush=True)
  return rec


async def main_async(a) -> None:
  R = Release(a.release)
  print(f"release verified: {R.verify()} files", flush=True)
  case_ids = a.cases or sorted(R.cases, key=lambda c: int(c[1:]))
  out = Path(a.out)
  cache = out / "cache"
  cache.mkdir(parents=True, exist_ok=True)
  client = AsyncOpenAI(timeout=TIMEOUT_S)
  sem = asyncio.Semaphore(a.concurrency)
  print(f"{len(case_ids)} scenarios x {len(a.models)} models x {a.seeds} seeds, word cap {a.word_cap or 'none'}", flush=True)

  async def one(case_id, model, seed):
    try:
      return await generate_one(client, sem, R, case_id, model, seed, a, cache)
    except Exception as exc:  # noqa: BLE001
      print(f"  FAILED {case_id} {model} s{seed}: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
      return None

  recs = await asyncio.gather(*(one(c, m, s) for m in a.models for c in case_ids for s in range(a.seeds)))
  done = [r for r in recs if r]
  write_jsonl(out / "briefs.jsonl", done)
  meta = {
    "release": str(R.root),
    "models": a.models,
    "condition": a.condition,
    "seeds": a.seeds,
    "scenarios": case_ids,
    "word_cap": a.word_cap,
    "reasoning_effort": a.reasoning_effort,
    "other_reasoning_effort": a.other_reasoning_effort,
    "extra_body": a.extra_body,
    "briefs": len(done),
    "failed": len(recs) - len(done),
    "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  }
  (out / "generate_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
  print(f"wrote {out / 'briefs.jsonl'} ({len(done)} briefs, {len(recs) - len(done)} failed)", flush=True)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--release", required=True, help="local copy of the MA-Bench dataset repository")
  ap.add_argument("--models", nargs="+", required=True, help="model ids as the endpoint expects them")
  ap.add_argument("--names", nargs="*", default=[], help="display names, model=name pairs (default: the model id)")
  ap.add_argument("--seeds", type=int, default=3)
  ap.add_argument("--cases", nargs="*", help="subset of case ids, e.g. C1 C3 (default: all ten)")
  ap.add_argument("--out", required=True)
  ap.add_argument("--condition", default=None, help="label for brief ids (default: benchmark_capped, or uncapped with --word-cap 0)")
  ap.add_argument("--word-cap", type=int, default=PROTOCOL_WORD_CAP, help="0 = uncapped ablation (no cap paragraph, no cut)")
  ap.add_argument("--reasoning-effort", default=None, help="reasoning_effort for OpenAI-style reasoning models (protocol: medium)")
  ap.add_argument("--other-reasoning-effort", default=None, help="reasoning_effort for other models where the provider exposes it")
  ap.add_argument("--extra-body", default=None, help='JSON of provider-specific request fields, e.g. \'{"chat_template_kwargs": {...}}\'')
  ap.add_argument("--concurrency", type=int, default=6)
  a = ap.parse_args()
  if not os.environ.get("OPENAI_BASE_URL") and not os.environ.get("OPENAI_API_KEY"):
    sys.exit("set OPENAI_BASE_URL and OPENAI_API_KEY for an OpenAI-compatible endpoint")
  a.condition = a.condition or ("benchmark_capped" if a.word_cap else "uncapped")
  a.names = dict(n.split("=", 1) for n in a.names)
  asyncio.run(main_async(a))


if __name__ == "__main__":
  main()
