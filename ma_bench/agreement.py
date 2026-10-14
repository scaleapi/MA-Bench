"""Agreement statistics for MA-Bench: the judge against the expert panel, and judges against each other.

  python -m ma_bench.agreement experts --release MA-Bench
    Judge vs the expert majority (ties dropped): agreement, Cohen's kappa, F1 with Fail as the positive class, overall
    and by severity. Fleiss' kappa for the four experts, and with the judge added as a fifth rater. Leave-one-out:
    the judge and each held-out expert against the majority of the other three.

  python -m ma_bench.agreement judges --release MA-Bench
    On the benchmark briefs every listed judge graded: item agreement and Cohen's kappa (pass/fail on every rubric
    item), Spearman correlation of model means, and each judge's self-preference excess on its own provider's models.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import statistics as st
import sys

from ma_bench.release import Release, load_jsonl, parse_brief_id
from ma_bench.score import brief_scores

FAMILIES = {  # judge provider -> model_slug prefixes of that provider's models
  "openai": ("openai_",),
  "azure_ai": ("openai_",),
  "gemini": ("gemini_", "google_"),
  "google": ("gemini_", "google_"),
  "vertex_ai": ("gemini_", "google_"),
  "anthropic": ("anthropic_",),
}


def cohen(a: list[str], b: list[str]) -> float:
  n = len(a)
  po = sum(x == y for x, y in zip(a, b)) / n
  pe = sum((a.count(c) / n) * (b.count(c) / n) for c in set(a) | set(b))
  return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def fleiss(ratings: list[list[str]]) -> float:
  """ratings: per item, the labels from each rater (same number of raters per item)."""
  n, N = len(ratings[0]), len(ratings)
  cats = sorted({c for r in ratings for c in r})
  p_j = {c: sum(r.count(c) for r in ratings) / (N * n) for c in cats}
  P_bar = st.mean((sum(r.count(c) ** 2 for c in cats) - n) / (n * (n - 1)) for r in ratings)
  P_e = sum(v * v for v in p_j.values())
  return (P_bar - P_e) / (1 - P_e) if P_e < 1 else float("nan")


def f1(pred: list[str], gold: list[str], pos: str = "fail") -> float:
  tp = sum(p == pos and g == pos for p, g in zip(pred, gold))
  fp = sum(p == pos and g != pos for p, g in zip(pred, gold))
  fn = sum(p != pos and g == pos for p, g in zip(pred, gold))
  return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")


def majority(labels: list[str]) -> str | None:
  c = collections.Counter(labels)
  if c["fail"] == c["pass"]:
    return None
  return "fail" if c["fail"] > c["pass"] else "pass"


def experts(R: Release) -> None:
  rows = load_jsonl(R.root / "runs" / "expert_study.jsonl")
  ex_cols = sorted(k for k in rows[0] if k.startswith("expert_") and k[7:].isdigit())
  judge_col = next(k for k in rows[0] if k.startswith("judge_") and k != "judge_note")
  lab = lambda v: str(v).strip().lower()
  E = [[lab(r[c]) for c in ex_cols] for r in rows]
  J = [lab(r[judge_col]) for r in rows]
  keep = [i for i, e in enumerate(E) if majority(e)]
  maj = {i: majority(E[i]) for i in keep}
  g = [maj[i] for i in keep]
  p = [J[i] for i in keep]
  print(f"{len(rows)} item verdicts, {len(ex_cols)} experts; expert majority on {len(keep)} (ties dropped: {len(rows) - len(keep)})")
  print(f"judge ({judge_col}) vs expert majority: agreement {st.mean(x == y for x, y in zip(p, g)):.1%}, "
        f"Cohen kappa {cohen(p, g):.3f}, F1 (fail) {f1(p, g):.3f}")
  print("\n| severity | items | agreement | kappa | experts fail | judge fails |\n|---|---:|---:|---:|---:|---:|")
  for sev in ("CI", "I", "SI", "D", "CD"):
    ids = [i for i in keep if rows[i]["severity"] == sev]
    if ids:
      gg, pp = [maj[i] for i in ids], [J[i] for i in ids]
      print(f"| {sev} | {len(ids)} | {st.mean(x == y for x, y in zip(pp, gg)):.0%} | {cohen(pp, gg):.2f} | "
            f"{st.mean(x == 'fail' for x in gg):.0%} | {st.mean(x == 'fail' for x in pp):.0%} |")
  print(f"\nFleiss kappa, {len(ex_cols)} experts: {fleiss(E):.3f}; with the judge as a fifth rater: "
        f"{fleiss([e + [j] for e, j in zip(E, J)]):.3f}")
  loo_judge, loo_expert = [], []
  for k in range(len(ex_cols)):
    m3 = [majority([e[x] for x in range(len(ex_cols)) if x != k]) for e in E]
    loo_judge.append(cohen(J, m3))
    loo_expert.append(cohen([e[k] for e in E], m3))
  print(f"leave-one-out vs the other experts' majority: judge mean kappa {st.mean(loo_judge):.3f}; held-out expert "
        f"mean {st.mean(loo_expert):.3f} (range {min(loo_expert):.2f} to {max(loo_expert):.2f})")
  jk = [cohen(J, [e[k] for e in E]) for k in range(len(ex_cols))]
  ek = [cohen([e[a] for e in E], [e[b] for e in E]) for a, b in itertools.combinations(range(len(ex_cols)), 2)]
  print(f"pairwise Cohen kappa: judge vs each expert mean {st.mean(jk):.3f}; expert vs expert mean {st.mean(ek):.3f}")


def average_ranks(values: list[float]) -> list[float]:
  """Ranks from 1, with tied values sharing their average rank."""
  order = sorted(range(len(values)), key=lambda i: values[i])
  ranks = [0.0] * len(values)
  i = 0
  while i < len(order):
    j = i
    while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
      j += 1
    for k in range(i, j + 1):
      ranks[order[k]] = (i + j) / 2 + 1
    i = j + 1
  return ranks


def spearman(x: dict[str, float], y: dict[str, float]) -> float:
  """Pearson correlation of average ranks, so ties count as ties; nan with fewer than three points or a constant side."""
  ks = sorted(x)
  if len(ks) < 3:
    return float("nan")
  rx, ry = average_ranks([x[k] for k in ks]), average_ranks([y[k] for k in ks])
  mx, my = st.mean(rx), st.mean(ry)
  sxx = sum((a - mx) ** 2 for a in rx)
  syy = sum((b - my) ** 2 for b in ry)
  if sxx == 0 or syy == 0:
    return float("nan")
  return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (sxx * syy) ** 0.5


def judges(R: Release, judge_names: list[str] | None, condition: str) -> None:
  items = R.items_by_case()
  by_judge: dict[str, dict[str, dict]] = collections.defaultdict(dict)
  provider: dict[str, str] = {}
  total = collections.Counter()
  for v in load_jsonl(R.root / "runs" / "verdicts.jsonl"):
    total[v["judge"]] += 1
    if parse_brief_id(v["brief_id"])["condition"] == condition:
      by_judge[v["judge"]][v["brief_id"]] = v
      provider[v["judge"]] = v.get("judge_model", "").split("/")[0]
  names = judge_names or sorted(by_judge, key=lambda j: (-total[j], j))  # primary judge (most verdicts) first
  missing = [j for j in names if j not in by_judge]
  if missing or len(names) < 2:
    sys.exit(f"need two or more judges with {condition} verdicts; available: {', '.join(sorted(by_judge)) or 'none'}"
             + (f"; not found: {', '.join(missing)}" if missing else ""))
  common = set.intersection(*(set(by_judge[j]) for j in names))
  if not common:
    sys.exit(f"no {condition} briefs were graded by all of {', '.join(names)}")
  scen = sorted({by_judge[names[0]][b]["case_id"] for b in common}, key=lambda c: int(c[1:]))
  print(f"{len(common)} {condition} briefs graded by {', '.join(names)} (scenarios {', '.join(scen)})")

  def passes(v):
    failed = set(v["failed_items"])
    return {iid: iid not in failed for iid in items[v["case_id"]]}

  means = {}
  for j in names:
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for b in common:
      v = by_judge[j][b]
      per[parse_brief_id(b)["model_slug"]][v["case_id"]].append(brief_scores(items[v["case_id"]], set(v["failed_items"]))["weighted"])
    means[j] = {m: st.mean(st.mean(s) for s in cases.values()) for m, cases in per.items()}

  def own(j, model_slug):
    return model_slug.startswith(FAMILIES.get(provider[j], ("\0",)))

  print("\n| judges | item verdicts | agreement | Cohen kappa | Spearman (model means) | own-family excess (first / second) |\n|---|---:|---:|---:|---:|---:|")
  for A, B in itertools.combinations(names, 2):
    va, vb = [], []
    for b in sorted(common):
      pa, pb = passes(by_judge[A][b]), passes(by_judge[B][b])
      for iid in pa:
        va.append("pass" if pa[iid] else "fail")
        vb.append("pass" if pb[iid] else "fail")
    d = {m: means[A][m] - means[B][m] for m in means[A]}

    def excess(j, sign):
      mine = [d[m] for m in d if own(j, m)]
      rest = [d[m] for m in d if not own(j, m)]
      return sign * (st.mean(mine) - st.mean(rest)) if mine and rest else float("nan")

    print(f"| {A} vs {B} | {len(va):,} | {st.mean(x == y for x, y in zip(va, vb)):.1%} | {cohen(va, vb):.3f} | "
          f"{spearman(means[A], means[B]):.2f} | {excess(A, 1):+.3f} / {excess(B, -1):+.3f} |")
  print("\nModel means on these briefs (mean over scenarios):")
  for m in sorted(means[names[0]], key=lambda m: -means[names[0]][m]):
    print(f"  {m}: " + " / ".join(f"{means[j][m]:.3f}" for j in names))


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = ap.add_subparsers(dest="cmd", required=True)
  e = sub.add_parser("experts", help="judge vs the expert panel")
  e.add_argument("--release", required=True)
  j = sub.add_parser("judges", help="agreement between judges")
  j.add_argument("--release", required=True)
  j.add_argument("--judges", nargs="*", help="judge names as in verdicts.jsonl (default: all)")
  j.add_argument("--condition", default="benchmark_capped")
  a = ap.parse_args()
  R = Release(a.release)
  if a.cmd == "experts":
    experts(R)
  else:
    judges(R, a.judges, a.condition)


if __name__ == "__main__":
  main()
