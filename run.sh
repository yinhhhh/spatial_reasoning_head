#!/usr/bin/env bash
set -euo pipefail

# Main-table runs: 4 datasets x 4 methods = 16 experiments.
# Methods:
#   1) Baseline (LLaVA-1.5)
#   2) ScaleVis (scaling_vis)
#   3) AdaptVis (max_prob + hard)
#   4) Ours (margin + continuous / sigmoid)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_ROOT="${ROOT_DIR}/outputs/main_table_16runs_100"
mkdir -p "${OUT_ROOT}"

export TEST_MODE=True
export TEST_SAMPLE_COUNT=100
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_ENDPOINT=https://hf-mirror.com

run_case() {
  local name="$1"
  shift
  local out_dir="${OUT_ROOT}/${name}"
  mkdir -p "${out_dir}"
  echo "=================================================="
  echo "RUN ${name}"
  python3 "${ROOT_DIR}/main_aro.py" "$@" --download --output-dir "${out_dir}"
}

run_all_for_dataset() {
  local dataset="$1"
  local option="$2"
  local scale_weight="$3"
  local w1="$4"
  local w2="$5"
  local adaptvis_th="$6"
  local ours_th="$7"
  local ours_alpha="$8"
  local ours_img_scale="$9"

  # 1) Baseline
  run_case "${dataset}__baseline" \
    --dataset "${dataset}" --model-name llava1.5 --method base --option "${option}"

  # 2) ScaleVis
  run_case "${dataset}__scalevis" \
    --dataset "${dataset}" --model-name llava1.5 --method scaling_vis --option "${option}" \
    --weight "${scale_weight}"

  # 3) AdaptVis (max-prob / hard)
  run_case "${dataset}__adaptvis" \
    --dataset "${dataset}" --model-name llava1.5 --method adapt_vis --option "${option}" \
    --weight1 "${w1}" --weight2 "${w2}" --threshold "${adaptvis_th}" \
    --uncertainty-criterion max_prob --adapt-weighting hard \
    --uncertainty-token-window 1 --uncertainty-token-agg mean

  # 4) Ours (margin + continuous)
  run_case "${dataset}__ours_margin_continuous" \
    --dataset "${dataset}" --model-name llava1.5 --method adapt_vis --option "${option}" \
    --weight1 "${w1}" --weight2 "${w2}" --threshold "${ours_th}" \
    --uncertainty-criterion margin --adapt-weighting sigmoid \
    --adapt-alpha "${ours_alpha}" --adapt-span 0.1 \
    --uncertainty-token-window 3 --uncertainty-token-agg mean \
    --image-attn-scale "${ours_img_scale}"
}

# COCO / VG 4-dataset main table settings
run_all_for_dataset "COCO_QA_one_obj" "four" 1.2 0.5 1.2 0.30 0.08 6 1.0
run_all_for_dataset "COCO_QA_two_obj" "four" 1.2 0.5 1.2 0.30 0.12 6 1.3
run_all_for_dataset "VG_QA_one_obj" "six" 2.0 0.5 2.0 0.20 0.32 6 1.0
run_all_for_dataset "VG_QA_two_obj" "six" 2.0 0.5 2.0 0.20 0.24 6 1.0

echo "All 16 runs finished."
echo "Results root: ${OUT_ROOT}"
