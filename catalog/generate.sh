#!/usr/bin/env bash
#
# Renders entity.catalog.input.yaml into entity.datadog.yaml.
# Datadog ingests the result from the default branch, so commit it.

set -euo pipefail

CATALOG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$CATALOG_DIR/.." && pwd)"
INPUT="$REPO_ROOT/entity.catalog.input.yaml"
OUTPUT="$REPO_ROOT/entity.datadog.yaml"

MARKER="# AUTO-GENERATED from entity.catalog.input.yaml"
HEADER="$MARKER by ./catalog/generate.sh. DO NOT EDIT."

if ! command -v cue >/dev/null 2>&1; then
  echo "cue not found. Install it with: brew install cue" >&2
  exit 1
fi

if [ ! -f "$INPUT" ]; then
  echo "No $INPUT. Nothing to render." >&2
  exit 1
fi

# No MARKER means someone wrote it by hand; overwriting would discard that.
if [[ -f "$OUTPUT" && "$(head -1 "$OUTPUT")" != "$MARKER"* ]]; then
  echo "entity.datadog.yaml is hand-maintained and would be overwritten." >&2
  echo "Move its contents into entity.catalog.input.yaml and delete it," >&2
  echo "or leave both alone." >&2
  exit 1
fi

# vet first: export reports a bad input from wherever the renderer touched it,
# while vet names the field and line.
cue vet "$CATALOG_DIR/schema.cue" --path '"input"' "$INPUT"

# Render fully before writing, so a failure leaves the old file intact.
rendered="$(
  printf '%s\n' "$HEADER"
  cue export "$CATALOG_DIR"/*.cue --path '"input"' "$INPUT" \
    --expression datadog.output --out text
)"

printf '%s\n' "$rendered" >"$OUTPUT"
echo "entity.datadog.yaml"
