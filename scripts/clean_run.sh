#!/usr/bin/env bash
# Level-1 check in a clean environment.
#
#   bash scripts/clean_run.sh PYTHON WORKDIR [RAW_DIR ENCODED_DIR ACTIONS [DECOMPOSITION_CACHE]]
#
# PYTHON   interpreter of a fresh virtual environment with requirements.txt installed
# WORKDIR  empty folder; the repository is copied there without evidence/
# With the data arguments, stages 01-05 rebuild the model inputs from the raw files and
# 19_verify_rebuild.py compares them with the published hashes in evidence/; the saved
# core-model checkpoints are then reloaded on the rebuilt inputs and their forecasts are
# recomputed and compared (16_leakage_checks.py).
#   RAW_DIR              folder with prices/ and documents.jsonl
#   ENCODED_DIR          document_ids.json, semantic.npy, sentiment.npy, encoding_manifest.json
#   ACTIONS              reviewed action register (JSONL)
#   DECOMPOSITION_CACHE  optional folder with {Market}_origin_parts/ chunks; without it the
#                        causal decomposition is recomputed (3-5 h on a CPU)
set -euo pipefail
PY="$1"; WORK="$2"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$WORK"
echo "== $(date -Iseconds) copy repository"
rsync -a --exclude evidence --exclude '__pycache__' "$REPO"/ "$WORK/IESF-Net/" 2>/dev/null \
  || { mkdir -p "$WORK/IESF-Net"; (cd "$REPO" && tar --exclude=./evidence --exclude=__pycache__ -cf - .) | (cd "$WORK/IESF-Net" && tar -xf -); }
cd "$WORK/IESF-Net"
"$PY" -c "import sys, torch, numpy, PyEMD; print('python', sys.version.split()[0], '| torch', torch.__version__, '| numpy', numpy.__version__, '| executable', sys.executable)"

echo "== $(date -Iseconds) unit tests"
"$PY" -m pytest tests -q -p no:cacheprovider

echo "== $(date -Iseconds) synthetic end-to-end run"
"$PY" scripts/run_pipeline.py --config configs/smoke.yaml --synthetic --mock-encoders

if [ "$#" -ge 5 ]; then
  RAW="$3"; ENC="$4"; ACT="$5"; CACHE="${6:-}"
  SET="paths.raw_dir=$RAW paths.interim_dir=$WORK/rebuild/interim paths.processed_dir=$WORK/rebuild/processed paths.output_dir=$WORK/rebuild/outputs"
  echo "== $(date -Iseconds) rebuild of the model inputs from the raw files"
  "$PY" scripts/01_prepare_corpus.py --set $SET
  "$PY" scripts/02_encode_text.py --import-dir "$ENC" --set $SET
  "$PY" scripts/03_extract_pii.py --actions "$ACT" --set $SET
  if [ -n "$CACHE" ]; then
    mkdir -p "$WORK/rebuild/interim/decomposition"
    cp -r "$CACHE"/*_origin_parts "$WORK/rebuild/interim/decomposition/"
    echo "decomposition chunks taken from $CACHE (archives, grouping and components are rebuilt from them)"
  fi
  "$PY" scripts/04_causal_decomposition.py --set $SET
  "$PY" scripts/05_build_dataset.py --set $SET
  "$PY" scripts/19_verify_rebuild.py --evidence "$REPO/evidence" --set $SET
  echo "== $(date -Iseconds) saved checkpoints on the rebuilt inputs"
  mkdir -p "$WORK/rebuild/outputs"
  cp -r "$REPO/evidence/checkpoints" "$REPO/evidence/forecasts" "$WORK/rebuild/outputs/"
  "$PY" scripts/16_leakage_checks.py --skip-decomposition --set $SET
fi
echo "== $(date -Iseconds) done"
