#!/usr/bin/env bash
set -euo pipefail

# 16 experiments total:
# Per dataset (4 datasets), run 4 cases:
#   1) baseline_all_heads
#   2) new_best_all_heads
#   3) new_best_selected16_heads
#   4) new_best_other16_heads
#
# New-method uses best params per dataset from previous runs.
# Sample count is fixed to 100 via TEST_SAMPLE_COUNT.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${ROOT_DIR}/outputs/compare_plus_headsplit_16runs_100"
mkdir -p "${OUT_ROOT}"

export TEST_MODE=True
export TEST_SAMPLE_COUNT=100
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_ENDPOINT=https://hf-mirror.com

SELECTED_16_HEADS="0,2,8,9,10,11,13,15,16,18,22,24,25,27,30,31"
OTHER_16_HEADS="1,3,4,5,6,7,12,14,17,19,20,21,23,26,28,29"

declare -a DATASETS=(
  # dataset:option:w1:w2:baseline_th:new_best_th
  "COCO_QA_one_obj:four:0.5:1.2:0.3:0.12"
  "COCO_QA_two_obj:four:0.5:1.2:0.3:0.3"
  "VG_QA_one_obj:six:0.5:2.0:0.2:0.12"
  "VG_QA_two_obj:six:0.5:2.0:0.2:0.2"
)

run_baseline() {
  local dataset="$1"
  local option="$2"
  local w1="$3"
  local w2="$4"
  local threshold="$5"

  local out_dir="${OUT_ROOT}/${dataset}__baseline_all_heads"
  mkdir -p "${out_dir}"

  echo "=================================================="
  echo "RUN dataset=${dataset} tag=baseline_all_heads"
  echo "output_dir=${out_dir}"

  python3 "${ROOT_DIR}/main_aro.py" \
    --dataset "${dataset}" \
    --model-name llava1.5 \
    --method adapt_vis \
    --option "${option}" \
    --output-dir "${out_dir}" \
    --weight1 "${w1}" \
    --weight2 "${w2}" \
    --threshold "${threshold}" \
    --uncertainty-criterion max_prob \
    --adapt-weighting hard \
    --uncertainty-token-window 1 \
    --uncertainty-token-agg mean
}

run_new_best() {
  local dataset="$1"
  local option="$2"
  local w1="$3"
  local w2="$4"
  local threshold="$5"
  local tag="$6"
  local active_heads="$7"

  local out_dir="${OUT_ROOT}/${dataset}__${tag}"
  mkdir -p "${out_dir}"

  echo "=================================================="
  echo "RUN dataset=${dataset} tag=${tag}"
  echo "active_heads=${active_heads:-<all_heads>}"
  echo "output_dir=${out_dir}"

  python3 "${ROOT_DIR}/main_aro.py" \
    --dataset "${dataset}" \
    --model-name llava1.5 \
    --method adapt_vis \
    --option "${option}" \
    --output-dir "${out_dir}" \
    --weight1 "${w1}" \
    --weight2 "${w2}" \
    --threshold "${threshold}" \
    --uncertainty-criterion margin \
    --adapt-weighting sigmoid \
    --adapt-alpha 8 \
    --adapt-span 0.1 \
    --uncertainty-token-window 3 \
    --uncertainty-token-agg mean \
    --active-heads "${active_heads}"
}

for item in "${DATASETS[@]}"; do
  dataset="${item%%:*}"
  rest="${item#*:}"
  option="${rest%%:*}"
  rest="${rest#*:}"
  w1="${rest%%:*}"
  rest="${rest#*:}"
  w2="${rest%%:*}"
  rest="${rest#*:}"
  baseline_th="${rest%%:*}"
  new_best_th="${rest##*:}"

  # 1) baseline all heads
  run_baseline "${dataset}" "${option}" "${w1}" "${w2}" "${baseline_th}"

  # 2) new method best params, all heads
  run_new_best "${dataset}" "${option}" "${w1}" "${w2}" "${new_best_th}" "new_best_all_heads" ""

  # 3) new method best params, selected 16 heads
  run_new_best "${dataset}" "${option}" "${w1}" "${w2}" "${new_best_th}" "new_best_selected16_heads" "${SELECTED_16_HEADS}"

  # 4) new method best params, other 16 heads
  run_new_best "${dataset}" "${option}" "${w1}" "${w2}" "${new_best_th}" "new_best_other16_heads" "${OTHER_16_HEADS}"
done

echo "All runs finished."
echo "Results root: ${OUT_ROOT}"
