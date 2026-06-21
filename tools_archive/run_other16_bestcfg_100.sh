#!/usr/bin/env bash
set -euo pipefail

# 8 runs total:
# - Re-run Controlled_Images_A/B with best margin configs on selected16 heads
# - Run all 6 datasets with best configs on other16 heads

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${ROOT_DIR}/outputs/other16_bestcfg_100"
mkdir -p "${OUT_ROOT}"

export TEST_MODE=True
export TEST_SAMPLE_COUNT=100
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_ENDPOINT=https://hf-mirror.com

SELECTED_16_HEADS="0,2,8,9,10,11,13,15,16,18,22,24,25,27,30,31"
OTHER_16_HEADS="1,3,4,5,6,7,12,14,17,19,20,21,23,26,28,29"

run_one() {
  local name="$1"
  local heads="$2"
  shift 2
  local out_dir="${OUT_ROOT}/${name}"
  mkdir -p "${out_dir}"
  echo "=================================================="
  echo "RUN ${name}"
  python3 "${ROOT_DIR}/main_aro.py" "$@" --output-dir "${out_dir}" --active-heads "${heads}"
}

# ---- Controlled A/B re-run: use best margin from 186-grid on selected16 ----
run_one "Controlled_Images_A__best_margin_selected16" "${SELECTED_16_HEADS}" \
  --dataset Controlled_Images_A --model-name llava1.5 --method adapt_vis --option four \
  --weight1 0.5 --weight2 1.5 --threshold 0.64 \
  --uncertainty-criterion margin --adapt-weighting sigmoid --adapt-alpha 8 --adapt-span 0.1 \
  --uncertainty-token-window 3 --uncertainty-token-agg mean

run_one "Controlled_Images_B__best_margin_selected16" "${SELECTED_16_HEADS}" \
  --dataset Controlled_Images_B --model-name llava1.5 --method adapt_vis --option four \
  --weight1 0.5 --weight2 1.5 --threshold 0.64 \
  --uncertainty-criterion margin --adapt-weighting sigmoid --adapt-alpha 6 --adapt-span 0.1 \
  --uncertainty-token-window 3 --uncertainty-token-agg mean

# ---- Other16 runs with each dataset's best config from 186-grid ----
run_one "Controlled_Images_A__bestcfg_other16" "${OTHER_16_HEADS}" \
  --dataset Controlled_Images_A --model-name llava1.5 --method adapt_vis --option four \
  --weight1 0.5 --weight2 1.5 --threshold 0.64 \
  --uncertainty-criterion margin --adapt-weighting sigmoid --adapt-alpha 8 --adapt-span 0.1 \
  --uncertainty-token-window 3 --uncertainty-token-agg mean

run_one "Controlled_Images_B__bestcfg_other16" "${OTHER_16_HEADS}" \
  --dataset Controlled_Images_B --model-name llava1.5 --method adapt_vis --option four \
  --weight1 0.5 --weight2 1.5 --threshold 0.64 \
  --uncertainty-criterion margin --adapt-weighting sigmoid --adapt-alpha 6 --adapt-span 0.1 \
  --uncertainty-token-window 3 --uncertainty-token-agg mean

run_one "COCO_QA_one_obj__bestcfg_other16" "${OTHER_16_HEADS}" \
  --dataset COCO_QA_one_obj --model-name llava1.5 --method adapt_vis --option four \
  --weight1 0.5 --weight2 1.2 --threshold 0.08 \
  --uncertainty-criterion margin --adapt-weighting sigmoid --adapt-alpha 6 --adapt-span 0.1 \
  --uncertainty-token-window 3 --uncertainty-token-agg mean

run_one "COCO_QA_two_obj__bestcfg_other16" "${OTHER_16_HEADS}" \
  --dataset COCO_QA_two_obj --model-name llava1.5 --method adapt_vis --option four \
  --weight1 0.5 --weight2 1.2 --threshold 0.12 \
  --uncertainty-criterion margin --adapt-weighting sigmoid --adapt-alpha 6 --adapt-span 0.1 \
  --uncertainty-token-window 3 --uncertainty-token-agg mean

run_one "VG_QA_one_obj__bestcfg_other16" "${OTHER_16_HEADS}" \
  --dataset VG_QA_one_obj --model-name llava1.5 --method adapt_vis --option six \
  --weight1 0.5 --weight2 2.0 --threshold 0.32 \
  --uncertainty-criterion margin --adapt-weighting sigmoid --adapt-alpha 6 --adapt-span 0.1 \
  --uncertainty-token-window 3 --uncertainty-token-agg mean

run_one "VG_QA_two_obj__bestcfg_other16" "${OTHER_16_HEADS}" \
  --dataset VG_QA_two_obj --model-name llava1.5 --method adapt_vis --option six \
  --weight1 0.5 --weight2 2.0 --threshold 0.24 \
  --uncertainty-criterion margin --adapt-weighting sigmoid --adapt-alpha 6 --adapt-span 0.1 \
  --uncertainty-token-window 3 --uncertainty-token-agg mean

python3 - <<'PY'
import json
from pathlib import Path

root = Path('/root/AdaptVis/outputs/other16_bestcfg_100')
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

echo "All runs finished."
