"""Grade MA-Bench briefs with the frozen judge.

One judge call per brief. The judge input is built only by the release's scoring/judge_input.py (task, case header and
the scenario's full rubric; the judge never sees the packet). The judge returns failed items only, as structured
output; every unlisted item passes. Calls are cached under <out>/judge, so an interrupted run resumes.

Environment: OPENAI_BASE_URL and OPENAI_API_KEY for an OpenAI-compatible endpoint that serves the judge model.

Example:
  python -m ma_bench.grade --release MA-Bench --briefs runs/my_run/briefs.jsonl --out runs/my_run \
    --judge openai/gpt-5.6
Writes <out>/verdicts_<judge>.jsonl in the release's runs/verdicts.jsonl layout, plus grade_meta.json.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import BaseModel

from ma_bench.release import Release, load_jsonl, parse_brief_id, sha256, slug, write_jsonl

DEFAULT_JUDGE = "openai/gpt-5.6"
TIMEOUT_S = 1800.0
JUDGE_RETRIES = 8
JUDGE_RETRY_WAIT_S = 45  # multiplied by the attempt number
JUDGE_MAX_TOKENS = 49152  # for providers that count hidden thinking against max_tokens


class JudgeFailure(BaseModel):
  id: str
  evidence: str
  rationale: str


class JudgeResponse(BaseModel):
  failures: list[JudgeFailure]
  overall_rationale: str


def judge_kwargs(judge: str, effort: str) -> dict:
  """The same effort for every judge. Providers that count hidden thinking against max_tokens get an explicit ceiling;
  OpenAI reasoning models manage their own budget."""
  if judge.startswith("openai/") or judge.startswith("azure_ai/gpt"):
    return {"reasoning_effort": effort}
  return {"reasoning_effort": effort, "max_tokens": JUDGE_MAX_TOKENS}


def cache_path(cache_dir: Path, brief_id: str) -> Path:
  """Cache file for a brief, built only from the validated parts of its id and kept inside cache_dir."""
  p = parse_brief_id(brief_id)  # raises on anything that is not <condition>/<scenario>__<model_slug>__s<seed>
  path = (cache_dir / f"{p['condition']}__{p['scenario']}__{p['model_slug']}__s{p['seed']}.json").resolve()
  if path.parent != cache_dir.resolve():
    raise ValueError(f"cache path escapes {cache_dir}: {brief_id!r}")
  return path


async def grade_one(client, sem, JI, B, brief: dict, judge: str, effort: str, valid: set[str], cache: Path) -> dict:
  system, user = JI.user_message(B, brief["case_id"], brief["text"])
  messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
  kwargs = judge_kwargs(judge, effort)
  # A cached verdict is reused only for the exact judge input (system prompt, case header, task, rubric and brief)
  # and the same judge settings; anything else is graded again rather than passing an old verdict off as current.
  fingerprint = hashlib.sha256(json.dumps([judge, kwargs, messages], sort_keys=True).encode()).hexdigest()
  if cache.exists():
    try:
      rec = json.loads(cache.read_text(encoding="utf-8"))
      if rec.get("fingerprint") == fingerprint:
        return rec
    except (json.JSONDecodeError, OSError):
      pass  # partial write from an interrupted run
    cache.unlink()
  async with sem:
    t0 = time.time()
    for attempt in range(JUDGE_RETRIES):
      try:
        resp = await client.chat.completions.parse(model=judge, messages=messages, response_format=JudgeResponse, **kwargs)
        break
      except Exception as exc:  # noqa: BLE001
        wait = JUDGE_RETRY_WAIT_S * (attempt + 1)
        print(f"  retry {attempt + 1} {brief['brief_id']}: {type(exc).__name__}: {str(exc)[:100]} (sleep {wait}s)", flush=True)
        await asyncio.sleep(wait)
    else:
      raise RuntimeError(f"judge failed repeatedly: {brief['brief_id']}")
  parsed = resp.choices[0].message.parsed
  if parsed is None:
    raise RuntimeError(f"unparsed judge output: {brief['brief_id']}")
  failures = [f.model_dump() for f in parsed.failures]
  rec = {
    "brief_id": brief["brief_id"],
    "judge": judge.split("/")[-1],
    "case_id": brief["case_id"],
    "failed_items": sorted({f["id"] for f in failures} & valid),
    "failures": failures,
    "unknown_ids": sorted({f["id"] for f in failures} - valid),
    "overall_rationale": parsed.overall_rationale,
    "judge_model": judge,
    "served_model": resp.model,
    "system_fingerprint": getattr(resp, "system_fingerprint", None),
    "usage": resp.usage.model_dump() if resp.usage else None,
    "fingerprint": fingerprint,
    "secs": round(time.time() - t0, 1),
  }
  cache.parent.mkdir(parents=True, exist_ok=True)
  tmp = cache.with_suffix(".json.tmp")
  tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
  tmp.replace(cache)
  print(f"  {brief['brief_id']:72s} failed={len(rec['failed_items']):3d} unknown={len(rec['unknown_ids'])} {rec['secs']:5.0f}s", flush=True)
  return rec


async def main_async(a) -> None:
  R = Release(a.release)
  n_verified = R.verify()
  JI = R.judge_input_module()
  B = JI.load_bundle(R.root)
  valid = {c: set(items) for c, items in R.items_by_case().items()}
  briefs = load_jsonl(a.briefs)
  bad = []
  for b in briefs:  # fail before any judge call, not halfway through a run
    try:
      parse_brief_id(b["brief_id"])
    except (KeyError, ValueError):
      bad.append(str(b.get("brief_id")))
    if b.get("case_id") not in valid:
      bad.append(f"{b.get('brief_id')} (unknown case_id {b.get('case_id')!r})")
  if bad:
    sys.exit(f"{len(bad)} briefs have an invalid brief_id or case_id, e.g. {bad[:3]}")
  if a.conditions:
    briefs = [b for b in briefs if b["condition"] in a.conditions]
  if a.limit:
    briefs = briefs[: a.limit]
  out = Path(a.out)
  meta = {
    "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "release": str(R.root),
    "release_files_verified": n_verified,
    "judge_prompt_sha256": sha256(R.root / "scoring" / "judge_prompt.md"),
    "rubric_items_sha256": sha256(R.root / "data" / "rubric_items.jsonl"),
    "briefs": str(a.briefs),
    "briefs_sha256": sha256(Path(a.briefs)),
    "judge": a.judge,
    "judge_kwargs": judge_kwargs(a.judge, a.effort),
    "n_briefs": len(briefs),
  }
  print(json.dumps(meta, indent=1), flush=True)
  if a.dry_run:
    return
  cache_dir = out / "judge" / slug(a.judge)
  cache_dir.mkdir(parents=True, exist_ok=True)
  client = AsyncOpenAI(timeout=TIMEOUT_S)
  sem = asyncio.Semaphore(a.concurrency)
  recs = await asyncio.gather(*[
    grade_one(client, sem, JI, B, b, a.judge, a.effort, valid[b["case_id"]], cache_path(cache_dir, b["brief_id"]))
    for b in briefs
  ])
  path = out / f"verdicts_{slug(a.judge)}.jsonl"
  write_jsonl(path, list(recs))
  meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
  meta["served_models"] = sorted({r["served_model"] for r in recs if r["served_model"]})
  meta["briefs_with_unknown_ids"] = sum(1 for r in recs if r["unknown_ids"])
  (out / "grade_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
  print(f"wrote {path} ({len(recs)} verdicts)", flush=True)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--release", required=True, help="local copy of the MA-Bench dataset repository")
  ap.add_argument("--briefs", required=True, help="briefs.jsonl from ma_bench.generate, or the release's runs/briefs.jsonl")
  ap.add_argument("--out", required=True)
  ap.add_argument("--judge", default=DEFAULT_JUDGE)
  ap.add_argument("--effort", default="high", help="reasoning effort for the judge (reference: high)")
  ap.add_argument("--conditions", nargs="*", help="only grade these conditions")
  ap.add_argument("--limit", type=int, default=0)
  ap.add_argument("--concurrency", type=int, default=6)
  ap.add_argument("--dry-run", action="store_true", help="verify the release and print the plan without calling the judge")
  a = ap.parse_args()
  if not Path(a.briefs).exists():
    sys.exit(f"no such file: {a.briefs}")
  asyncio.run(main_async(a))


if __name__ == "__main__":
  main()
