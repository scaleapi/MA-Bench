"""Read the MA-Bench release (huggingface.co/datasets/ScaleAI/MA-Bench) and build model inputs exactly as the
reference runs did.

The release is the only input. The task prompt, the length-cap paragraph and the packet layout come from
scoring/task_prompt.md and data/*.jsonl; the judge input comes from the release's own scoring/judge_input.py.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

PROTOCOL_WORD_CAP = 3000
WORD_RE = re.compile(r"\S*[A-Za-z0-9]\S*")  # a word is a whitespace-delimited token with a letter or digit
BRIEF_ID_RE = re.compile(r"(?P<condition>[a-z0-9_]+)/(?P<scenario>[A-Z]+-[A-Z]+)__(?P<model_slug>[a-z0-9_]+)__s(?P<seed>\d+)")


def load_jsonl(path: str | Path) -> list[dict]:
  with open(path, encoding="utf-8") as f:
    return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
  Path(path).parent.mkdir(parents=True, exist_ok=True)
  with open(path, "w", encoding="utf-8") as f:
    for r in rows:
      f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def slug(model: str) -> str:
  """Model id to the slug used in brief ids, e.g. "anthropic/claude-opus-5" -> "anthropic_claude_opus_5"."""
  return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def parse_brief_id(brief_id: str) -> dict:
  m = BRIEF_ID_RE.fullmatch(brief_id)
  if not m:
    raise ValueError(f"not a brief id: {brief_id!r}")
  d = m.groupdict()
  d["seed"] = int(d["seed"])
  return d


def cap_words(text: str, cap: int) -> tuple[str, int, bool]:
  """Cut after the cap-th word, keeping newlines and markdown. Returns (text, words before the cut, capped)."""
  ends = [m.end() for m in WORD_RE.finditer(text)]
  if cap and len(ends) > cap:
    return text[: ends[cap - 1]], len(ends), True
  return text, len(ends), False


class Release:
  def __init__(self, root: str | Path):
    self.root = Path(root)
    self.cases = {c["case_id"]: c for c in load_jsonl(self.root / "data" / "cases.jsonl")}
    self.documents = load_jsonl(self.root / "data" / "documents.jsonl")
    self.items = load_jsonl(self.root / "data" / "rubric_items.jsonl")
    blocks = re.findall(r"```\n(.*?)\n```", (self.root / "scoring" / "task_prompt.md").read_text(encoding="utf-8"), re.S)
    self.task_template, self.cap_paragraph = blocks[0], blocks[1]

  def verify(self, prefixes: tuple[str, ...] = ("data/", "scoring/")) -> int:
    """Check every release file under prefixes against MANIFEST.json. Exits on any mismatch."""
    manifest = json.loads((self.root / "MANIFEST.json").read_text(encoding="utf-8"))["files"]
    bad, n = [], 0
    for rel, meta in manifest.items():
      if not rel.startswith(prefixes):
        continue
      n += 1
      p = self.root / rel
      if not p.exists():
        bad.append(f"missing {rel}")
      elif sha256(p) != meta["sha256"]:
        bad.append(f"sha256 mismatch {rel}")
    if bad:
      sys.exit("release verification failed:\n  " + "\n  ".join(bad))
    return n

  def scenario_code(self, case_id: str) -> str:
    """Scenario code used in brief ids, e.g. Operation SILENT FATHOM, full packet -> "SF-FULL"."""
    c = self.cases[case_id]
    return "".join(w[0] for w in c["operation"].split()) + "-" + c["difficulty"].upper()

  def task(self, case_id: str) -> str:
    return self.task_template.replace("{op}", self.cases[case_id]["operation"])

  def packet(self, case_id: str) -> str:
    docs = sorted((d for d in self.documents if d["case_id"] == case_id), key=lambda d: d["doc_order"])
    head = f"# {self.cases[case_id]['operation']} — Mission Analysis Source Packet\n"
    return head + "".join(
      f"\n\n## Document {d['doc_number']} — {d['title']} (source file: {d['source_file']})\n\n{d['text']}\n" for d in docs
    )

  def model_input(self, case_id: str, word_cap: int = PROTOCOL_WORD_CAP) -> str:
    """The single user message a system under test receives. word_cap=0 is the uncapped ablation (no cap paragraph)."""
    cap = ""
    if word_cap:
      cap = "\n\n" + self.cap_paragraph.replace(f"{PROTOCOL_WORD_CAP:,}", f"{word_cap:,}")
    return f"{self.task(case_id)}{cap}\n\n# SOURCE DOCUMENTS\n\n{self.packet(case_id)}"

  def items_by_case(self) -> dict[str, dict[str, dict]]:
    by: dict[str, dict[str, dict]] = {}
    for it in self.items:
      by.setdefault(it["case_id"], {})[it["item_id"]] = it
    return by

  def judge_input_module(self):
    """The release's own judge-input builder (scoring/judge_input.py); no other construction is valid."""
    spec = importlib.util.spec_from_file_location("ma_bench_release_judge_input", self.root / "scoring" / "judge_input.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
