#!/usr/bin/env bash
set -euo pipefail

# 6 runs total: one per dataset.
# Uses best configs from outputs/grid_6ds_th008_064_a6a8_100/_summary_all186.json
# while enabling scaling only on the selected 16 heads.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${ROOT_DIR}/outputs/best6_selected16_100"
mkdir -p "${OUT_ROOT}"

export TEST_MODE=True
export TEST_SAMPLE_COUNT=100
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_ENDPOINT=https://hf-mirror.com

SELECTED_16_HEADS="0,2,8,9,10,11,13,15,16,18,22,24,25,27,30,31"

run_one() {
  local name="$1"
  shift
  local out_dir="${OUT_ROOT}/${name}"
  mkdir -p "${out_dir}"
  echo "=================================================="
  echo "RUN ${name}"
  python3 "${ROOT_DIR}/main_aro.py" "$@" --output-dir "${out_dir}" --active-heads "${SELECTED_16_HEADS}"
}

# Controlled_Images_A: best was baseline (hard max_prob, th=0.4)
run_one "Controlled_Images_A__bestcfg_selected16" \
  --dataset Controlled_Images_A --model-name llava1.5 --method adapt_vis --option four \
  --weight1 0.5 --weight2 1.5 --threshold 0.4 \
  --uncertainty-criterion max_prob --adapt-weighting hard \
  --uncertainty-token-window 1 --uncertainty-token-agg mean

# Controlled_Images_B: best was baseline (hard max_prob, th=0.35)
run_one "Controlled_Images_B__bestcfg_selected16" \
  --dataset Controlled_Images_B --model-name llava1.5 --method adapt_vis --option four \
  --weight1 0.5 --weight2 1.5 --threshold 0.35 \
  --uncertainty-criterion max_prob --adapt-weighting hard \
  --uncertainty-token-window 1 --uncertainty-token-agg mean

# COCO_QA_one_obj: best grid case a6_th0.08 (margin+sigmoid)
run_one "COCO_QA_one_obj__bestcfg_selected16" \
  --dataset COCO_QA_one_obj --model-name llava1.5 --method adapt_vis --option four \
  --weight1 0.5 --weight2 1.2 --threshold 0.08 \
  --uncertainty-criterion margin --adapt-weighting sigmoid --adapt-alpha 6 --adapt-span 0.1 \
  --uncertainty-token-window 3 --uncertainty-token-agg mean

# COCO_QA_two_obj: best grid case a6_th0.12 (margin+sigmoid)
run_one "COCO_QA_two_obj__bestcfg_selected16" \
  --dataset COCO_QA_two_obj --model-name llava1.5 --method adapt_vis --option four \
  --weight1 0.5 --weight2 1.2 --threshold 0.12 \
  --uncertainty-criterion margin --adapt-weighting sigmoid --adapt-alpha 6 --adapt-span 0.1 \
  --uncertainty-token-window 3 --uncertainty-token-agg mean

# VG_QA_one_obj: best grid case a6_th0.32 (margin+sigmoid)
run_one "VG_QA_one_obj__bestcfg_selected16" \
  --dataset VG_QA_one_obj --model-name llava1.5 --method adapt_vis --option six \
  --weight1 0.5 --weight2 2.0 --threshold 0.32 \
  --uncertainty-criterion margin --adapt-weighting sigmoid --adapt-alpha 6 --adapt-span 0.1 \
  --uncertainty-token-window 3 --uncertainty-token-agg mean

# VG_QA_two_obj: best grid case a6_th0.24 (margin+sigmoid)
run_one "VG_QA_two_obj__bestcfg_selected16" \
  --dataset VG_QA_two_obj --model-name llava1.5 --method adapt_vis --option six \
  --weight1 0.5 --weight2 2.0 --threshold 0.24 \
  --uncertainty-criterion margin --adapt-weighting sigmoid --adapt-alpha 6 --adapt-span 0.1 \
  --uncertainty-token-window 3 --uncertainty-token-agg mean

python3 - <<'PY'
import json
from pathlib import Path

root = Path('/root/AdaptVis/outputs/best6_selected16_100')
rows = []
for p in sorted(root.glob('*/res.json')):
    lines = [x for x in p.read_text().splitlines() if x.strip()]
    if not lines:
        continue
    j = json.loads(lines[-1])
    rows.append({
        'run': p.parent.name,
        'dataset': j.get('dataset'),
        'individual_accuracy': j.get('Individual accuracy'),
        'pair_accuracy': j.get('Pair accuracy'),
        'set_accuracy': j.get('Set accuracy'),
        'correct_count': len(j.get('correct_id', [])) if isinstance(j.get('correct_id'), list) else None,
    })
summary_path = root / 'summary.json'
summary_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
print('SUMMARY_PATH', summary_path)
PY

echo "All 6 runs finished."
