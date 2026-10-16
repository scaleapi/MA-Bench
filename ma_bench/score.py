"""Score MA-Bench verdicts: per-brief scores, then per condition x model means with 95% bootstrap intervals.

A model's score is the mean over scenarios of its mean brief score. Intervals resample scenarios with replacement
(2,000 draws, seed 0), with seeds averaged within a scenario.

Weighted = (positive weight earned + negative weight incurred) / total positive weight; not clamped.
Unweighted = share of items correct, where a negative item is correct when the mistake it names is absent.

Example (the paper's numbers from the released verdicts):
  python -m ma_bench.score --release MA-Bench --judge gpt-5.6
Example (your own run):
  python -m ma_bench.score --release MA-Bench --verdicts runs/my_run/verdicts_openai_gpt_5_6.jsonl \
    --briefs runs/my_run/briefs.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import statistics as st
from pathlib import Path

from ma_bench.release import Release, load_jsonl, parse_brief_id

CONDITION_ORDER = ["benchmark_capped", "ladder_capped", "effort_low", "effort_high", "uncapped"]


def brief_scores(items: dict[str, dict], failed: set[str]) -> dict:
  """failed: item ids the judge listed. An unlisted negative item means its mistake is present."""
  pos = [i for i in items.values() if i["weight"] > 0]
  neg = [i for i in items.values() if i["weight"] < 0]
  earned = sum(i["weight"] for i in pos if i["item_id"] not in failed)
  incurred = sum(i["weight"] for i in neg if i["item_id"] not in failed)
  correct = sum(1 for i in pos if i["item_id"] not in failed) + sum(1 for i in neg if i["item_id"] in failed)
  return {
    "weighted": (earned + incurred) / sum(i["weight"] for i in pos),
    "unweighted": correct / len(items),
    "positive_pass_rate": sum(1 for i in pos if i["item_id"] not in failed) / len(pos),
    "mistakes_made": sum(1 for i in neg if i["item_id"] not in failed),
    "critical_mistakes": sum(1 for i in neg if i["item_id"] not in failed and i["severity"] == "CD"),
  }


def bootstrap(per_scenario: dict[str, list[float]], n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
  """Mean of scenario means, with a percentile interval from resampling scenarios."""
  rng = random.Random(seed)
  scen = list(per_scenario)
  means = {s: st.mean(v) for s, v in per_scenario.items()}
  point = st.mean(means.values())
  draws = sorted(st.mean(means[s] for s in rng.choices(scen, k=len(scen))) for _ in range(n))
  return point, draws[int(0.025 * n)], draws[int(0.975 * n)]


def score_rows(R: Release, verdicts: list[dict]) -> list[dict]:
  items = R.items_by_case()
  rows = []
  for v in verdicts:
    meta = parse_brief_id(v["brief_id"])
    case_items = items[v["case_id"]]
    failed = set(v["failed_items"]) & set(case_items)
    rows.append({"brief_id": v["brief_id"], "case_id": v["case_id"], **meta, **brief_scores(case_items, failed)})
  return rows


def group_scores(rows: list[dict], names: dict[str, str]) -> list[dict]:
  groups = collections.defaultdict(lambda: collections.defaultdict(list))
  for r in rows:
    groups[(r["condition"], r["model_slug"])][r["scenario"]].append(r)
  out = []
  for (cond, model_slug), per in groups.items():
    rec = {
      "condition": cond,
      "model": names.get(model_slug, model_slug),
      "model_slug": model_slug,
      "n_briefs": sum(len(v) for v in per.values()),
      "n_scenarios": len(per),
    }
    for metric in ("weighted", "unweighted", "positive_pass_rate"):
      p, lo, hi = bootstrap({s: [x[metric] for x in v] for s, v in per.items()})
      rec[metric] = {"mean": round(p, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    rec["mistakes_per_brief"] = round(st.mean(x["mistakes_made"] for v in per.values() for x in v), 2)
    rec["critical_mistakes_per_brief"] = round(st.mean(x["critical_mistakes"] for v in per.values() for x in v), 2)
    out.append(rec)
  order = {c: i for i, c in enumerate(CONDITION_ORDER)}
  out.sort(key=lambda r: (order.get(r["condition"], len(order)), r["condition"], -r["weighted"]["mean"]))
  return out


def markdown(groups: list[dict]) -> str:
  md = [
    "| condition | model | briefs | weighted score (95% CI) | items correct (95% CI) | positive pass rate | mistakes / brief (critical) |",
    "|---|---|---:|---|---|---:|---:|",
  ]
  for r in groups:
    w, u = r["weighted"], r["unweighted"]
    model = r["model"] + (f" (partial, {r['n_scenarios']} scenarios)" if r.get("partial") else "")
    md.append(
      f"| {r['condition']} | {model} | {r['n_briefs']} | {w['mean']:.3f} ({w['ci95'][0]:.3f} to {w['ci95'][1]:.3f}) | "
      f"{u['mean']:.1%} ({u['ci95'][0]:.1%} to {u['ci95'][1]:.1%}) | {r['positive_pass_rate']['mean']:.1%} | "
      f"{r['mistakes_per_brief']} ({r['critical_mistakes_per_brief']}) |"
    )
  return "\n".join(md)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--release", required=True, help="local copy of the MA-Bench dataset repository")
  ap.add_argument("--verdicts", help="verdicts JSONL (default: the release's runs/verdicts.jsonl)")
  ap.add_argument("--briefs", help="briefs JSONL for display names (default: the release's runs/briefs.jsonl)")
  ap.add_argument("--judge", help="keep only this judge's verdicts, e.g. gpt-5.6 (required if the file has several)")
  ap.add_argument("--conditions", nargs="*")
  ap.add_argument("--allow-partial", action="store_true",
                  help="score models that lack some scenarios (marked partial) instead of stopping")
  ap.add_argument("--md")
  ap.add_argument("--json")
  a = ap.parse_args()
  R = Release(a.release)
  verdicts = load_jsonl(a.verdicts or R.root / "runs" / "verdicts.jsonl")
  judges = sorted({v["judge"] for v in verdicts})
  if a.judge:
    verdicts = [v for v in verdicts if v["judge"] == a.judge]
  elif len(judges) > 1:
    ap.error(f"the verdicts come from several judges ({', '.join(judges)}); pick one with --judge")
  if a.conditions:
    verdicts = [v for v in verdicts if parse_brief_id(v["brief_id"])["condition"] in a.conditions]
  briefs_path = Path(a.briefs) if a.briefs else R.root / "runs" / "briefs.jsonl"
  names = {b["model_slug"]: b["model"] for b in load_jsonl(briefs_path)} if briefs_path.exists() else {}
  rows = score_rows(R, verdicts)
  groups = group_scores(rows, names)
  # A model's score is a mean over scenarios, so a model missing scenarios is scored on a different benchmark.
  # Stop by default; with --allow-partial, score it and mark it.
  n_all = len(R.cases)
  partial = [g for g in groups if g["n_scenarios"] < n_all]
  for g in partial:
    g["partial"] = True
  if partial and not a.allow_partial:
    ap.exit(1, "models missing scenarios (scores would not be comparable): "
            + "; ".join(f"{g['condition']} {g['model']} {g['n_scenarios']}/{n_all}" for g in partial)
            + "\nrerun generation and grading to fill the gaps, or pass --allow-partial\n")
  text = markdown(groups)
  print(text)
  if a.md:
    Path(a.md).write_text(text + "\n", encoding="utf-8")
  if a.json:
    Path(a.json).write_text(json.dumps({"briefs": rows, "groups": groups}, indent=1), encoding="utf-8")


if __name__ == "__main__":
  main()
