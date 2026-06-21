#!/usr/bin/env bash
set -euo pipefail

# Grid setup:
# - 6 datasets
# - alpha in {6, 8}
# - threshold from 0.08 to 0.64 (step 0.04)
# - TEST_SAMPLE_COUNT=100
#
# Total grid runs: 6 * 2 * 15 = 180
# Optional baseline runs: +6 (set RUN_BASELINE=true)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${ROOT_DIR}/outputs/grid_6ds_th008_064_a6a8_100"
mkdir -p "${OUT_ROOT}"

export TEST_MODE=True
export TEST_SAMPLE_COUNT=100
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_ENDPOINT=https://hf-mirror.com

# If you want baseline runs too, set to "true".
RUN_BASELINE="${RUN_BASELINE:-false}"

# dataset:option:w1:w2:baseline_th
declare -a DATASETS=(
  "Controlled_Images_A:four:0.5:1.5:0.4"
  "Controlled_Images_B:four:0.5:1.5:0.35"
  "COCO_QA_one_obj:four:0.5:1.2:0.3"
  "COCO_QA_two_obj:four:0.5:1.2:0.3"
  "VG_QA_one_obj:six:0.5:2.0:0.2"
  "VG_QA_two_obj:six:0.5:2.0:0.2"
)

declare -a ALPHAS=(6 8)
declare -a THS=(
  0.08 0.12 0.16 0.20 0.24 0.28 0.32 0.36
  0.40 0.44 0.48 0.52 0.56 0.60 0.64
)

run_baseline() {
  local dataset="$1"
  local option="$2"
  local w1="$3"
  local w2="$4"
  local baseline_th="$5"
  local out_dir="${OUT_ROOT}/${dataset}__baseline"
  mkdir -p "${out_dir}"

  echo "=================================================="
  echo "RUN baseline dataset=${dataset}"
  python3 "${ROOT_DIR}/main_aro.py" \
    --dataset "${dataset}" \
    --model-name llava1.5 \
    --method adapt_vis \
    --option "${option}" \
    --output-dir "${out_dir}" \
    --weight1 "${w1}" \
    --weight2 "${w2}" \
    --threshold "${baseline_th}" \
    --uncertainty-criterion max_prob \
    --adapt-weighting hard \
    --uncertainty-token-window 1 \
    --uncertainty-token-agg mean
}

run_grid_one() {
  local dataset="$1"
  local option="$2"
  local w1="$3"
  local w2="$4"
  local alpha="$5"
  local th="$6"
  local out_dir="${OUT_ROOT}/${dataset}__a${alpha}_th${th}"
  mkdir -p "${out_dir}"

  echo "RUN grid dataset=${dataset} alpha=${alpha} th=${th}"
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
    --adapt-alpha "${alpha}" \
    --adapt-span 0.1 \
    --uncertainty-token-window 3 \
    --uncertainty-token-agg mean
}

total_grid=$(( ${#DATASETS[@]} * ${#ALPHAS[@]} * ${#THS[@]} ))
echo "Planned grid runs: ${total_grid}"
if [[ "${RUN_BASELINE}" == "true" ]]; then
  echo "Baseline runs enabled: +${#DATASETS[@]}"
fi
echo "Output root: ${OUT_ROOT}"

for item in "${DATASETS[@]}"; do
  dataset="${item%%:*}"
  rest="${item#*:}"
  option="${rest%%:*}"
  rest="${rest#*:}"
  w1="${rest%%:*}"
  rest="${rest#*:}"
  w2="${rest%%:*}"
  baseline_th="${rest##*:}"

  if [[ "${RUN_BASELINE}" == "true" ]]; then
    run_baseline "${dataset}" "${option}" "${w1}" "${w2}" "${baseline_th}"
  fi

  for alpha in "${ALPHAS[@]}"; do
    for th in "${THS[@]}"; do
      run_grid_one "${dataset}" "${option}" "${w1}" "${w2}" "${alpha}" "${th}"
    done
  done
done

echo "All runs finished."
