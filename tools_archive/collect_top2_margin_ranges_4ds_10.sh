#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${ROOT_DIR}/outputs/top2_margin_ranges_4ds_10"
mkdir -p "${OUT_ROOT}"

export TEST_MODE=True
export TEST_SAMPLE_COUNT=10
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_ENDPOINT=https://hf-mirror.com
export LOG_TOP2_MARGIN=1

# dataset:option:w1:w2:th(best)
declare -a DATASETS=(
  "COCO_QA_one_obj:four:0.5:1.2:0.12"
  "COCO_QA_two_obj:four:0.5:1.2:0.3"
  "VG_QA_one_obj:six:0.5:2.0:0.12"
  "VG_QA_two_obj:six:0.5:2.0:0.2"
)

for item in "${DATASETS[@]}"; do
  dataset="${item%%:*}"
  rest="${item#*:}"
  option="${rest%%:*}"
  rest="${rest#*:}"
  w1="${rest%%:*}"
  rest="${rest#*:}"
  w2="${rest%%:*}"
  th="${rest##*:}"

  out_dir="${OUT_ROOT}/${dataset}"
  log_file="${OUT_ROOT}/${dataset}.log"
  mkdir -p "${out_dir}"

  echo "RUN dataset=${dataset}"
  python3 "${ROOT_DIR}/main_aro.py" \
    --dataset "${dataset}" \
    --model-name llava1.5 \
    --method adapt_vis \
    --option "${option}" \
    --output-dir "${out_dir}" \
    --weight1 "${w1}" \
    --weight2 "${w2}" \
    --threshold "${th}" \
    --uncertainty-criterion margin \
    --adapt-weighting sigmoid \
    --adapt-alpha 8 \
    --adapt-span 0.1 \
    --uncertainty-token-window 3 \
    --uncertainty-token-agg mean \
    > "${log_file}" 2>&1
done

python3 - <<'PY'
import json
import re
from pathlib import Path

root = Path("/root/AdaptVis/outputs/top2_margin_ranges_4ds_10")
pat = re.compile(r"\[top2_debug\]\s+top1=([0-9]*\.?[0-9]+)\s+top2=([0-9]*\.?[0-9]+)\s+margin=([0-9]*\.?[0-9]+)")

def q(vals, p):
    if not vals:
        return None
    idx = int(round((len(vals) - 1) * p))
    idx = max(0, min(idx, len(vals) - 1))
    return vals[idx]

summary = []
for log in sorted(root.glob("*.log")):
    dataset = log.stem
    top1s, top2s, margins = [], [], []
    for line in log.read_text().splitlines():
        m = pat.search(line)
        if not m:
            continue
        top1s.append(float(m.group(1)))
        top2s.append(float(m.group(2)))
        margins.append(float(m.group(3)))

    top1s.sort()
    top2s.sort()
    margins.sort()
    rec = {
        "dataset": dataset,
        "n_samples_captured": len(margins),
        "top1": {
            "min": min(top1s) if top1s else None,
            "p50": q(top1s, 0.5),
            "max": max(top1s) if top1s else None,
        },
        "top2": {
            "min": min(top2s) if top2s else None,
            "p50": q(top2s, 0.5),
            "max": max(top2s) if top2s else None,
        },
        "margin": {
            "min": min(margins) if margins else None,
            "p25": q(margins, 0.25),
            "p50": q(margins, 0.5),
            "p75": q(margins, 0.75),
            "max": max(margins) if margins else None,
        },
    }
    summary.append(rec)

summary_path = root / "summary.json"
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(summary_path)
PY

echo "Done. Summary at: ${OUT_ROOT}/summary.json"
