# AdaptVis Spatial Reasoning Experiments

This repository contains the code, analysis scripts, and report assets for our NeurIPS-style write-up on head-level spatial reasoning in VLMs.

Repository URL: [https://github.com/yinhhhh/spatial_reasoning_head](https://github.com/yinhhhh/spatial_reasoning_head)

## What This Repo Reproduces

The report follows a mechanism-to-method story:

1. **Head diagnosis**: measure per-head YOLO-region overlap and overlap-to-correctness AUROC.
2. **Naive intervention check**: direct in-box boosting is unstable.
3. **Improved method**: margin-guided continuous attention reallocation for better spatial reasoning.

## Environment Setup

```bash
git clone https://github.com/yinhhhh/spatial_reasoning_head.git
cd spatial_reasoning_head
python3 -m pip install -r requirements.txt
mkdir -p data output outputs
```

Optional (if your environment needs mirror endpoints):

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_ENDPOINT=https://hf-mirror.com
```

## Data

- Dataset loading and evaluation logic: `dataset_zoo/aro_datasets.py`
- QA prompts: `prompt/`
- `main_aro.py --download` will auto-download required dataset files if missing.

## Core Experiment Commands

Set 100-sample test mode:

```bash
export TEST_MODE=True
export TEST_SAMPLE_COUNT=100
```

Run a baseline example:

```bash
python3 main_aro.py \
  --dataset Controlled_Images_A \
  --model-name llava1.5 \
  --download \
  --method base \
  --option four \
  --output-dir outputs/control_image_a_base_100
```

Run an AdaptVis-style baseline configuration:

```bash
python3 main_aro.py \
  --dataset Controlled_Images_A \
  --model-name llava1.5 \
  --download \
  --method adapt_vis \
  --weight1 0.5 \
  --weight2 1.5 \
  --threshold 0.4 \
  --option four \
  --output-dir outputs/control_image_a_adaptvis_100
```

Batch scripts used in our experiments:

- `run.sh`
- `tools/run_newmethod_headsplit_100.sh`
- `tools/run_best6_selected16_100.sh`
- `tools/run_best6_allhead_image12_100.sh`
- `tools/run_other16_bestcfg_100.sh`

## Multi-Head YOLO Overlap and AUROC

General 4-dataset analyzer:

```bash
python3 tools/analyze_head_overlap_auroc.py \
  --dataset-name Controlled_Images_A \
  --attn-dir output/Controlled_Images_A_weight1.00 \
  --result-jsonl outputs/control_image_a_adaptvis_100/res.json \
  --sample-count 100 \
  --layer 17 \
  --yolo-model yolov8n.pt \
  --output-dir outputs/head_verify_margin_continuous/control_a_margin_sigmoid_head_analysis \
  --title-prefix Controlled_Images_A
```

Controlled_Images_A dedicated analyzer:

```bash
python3 tools/analyze_control_image_a_heads.py \
  --attn-dir output/Controlled_Images_A_weight1.00 \
  --result-jsonl outputs/control_image_a_adaptvis_100/res.json \
  --sample-count 100 \
  --layer 17 \
  --yolo-model yolov8n.pt \
  --output-dir outputs/head_verify_4runs/control_a_baseline_head_analysis
```

The scripts export:

- `head_overlap_bar.png`
- `head_auroc_bar.png`
- `summary.json`

## Report Build

```bash
cd report
latexmk main.tex
```

Main report files:

- `report/main.tex`
- `report/references.bib`
- `report/figures/`

## Source Code Reference

For code review and method tracing, start from:

- `main_aro.py` (entry point)
- `model_zoo/llava15.py` (model wrapper / decoding flow)
- `model_zoo/llava/modeling_llava_scal.py` (image attention intervention path)
- `model_zoo/llama/modeling_llama_add_attn.py` (attention/head-level modifications)
- `tools/analyze_head_overlap_auroc.py` (per-head overlap + AUROC)
- `tools/eval_yolo_attention_overlap.py` (YOLO overlap comparison and visual diagnostics)

## Base Acknowledgement

This project is built on top of prior open-source work from:

- What's "up" with vision-language models? [[paper](https://arxiv.org/pdf/2310.19785)] [[code](https://github.com/amitakamath/whatsup_vlms)]

