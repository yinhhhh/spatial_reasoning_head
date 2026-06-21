# CV Course Project: Head-Level Spatial Reasoning in VLMs

This repository contains the code, analysis scripts, and report assets for our computer vision course project on head-level spatial reasoning in VLMs.

Repository URL: [https://github.com/yinhhhh/spatial_reasoning_head](https://github.com/yinhhhh/spatial_reasoning_head)

We strongly recommend you to read our report first.

## Acknowledgement

This project is based on the AdaptVis codebase. We thank the original authors for releasing their implementation:

- AdaptVis original project: [https://github.com/shiqichen17/AdaptVis](https://github.com/shiqichen17/AdaptVis)

## What This Repo Reproduces

The report follows a mechanism-to-method story:

1. **Head diagnosis**: measure per-head YOLO-region overlap and overlap-to-correctness AUROC.
2. **Improved method**: margin-guided continuous attention reallocation for better spatial reasoning.

## Data

- `main_aro.py --download` will auto-download required dataset files if missing.

- Alternatively, you can manually download data in one of the two ways:
  - Google Drive: [https://drive.google.com/drive/u/0/folders/164q6X9hrvP-QYpi3ioSnfMuyHpG5oRkZ](https://drive.google.com/drive/u/0/folders/164q6X9hrvP-QYpi3ioSnfMuyHpG5oRkZ)
  - Hugging Face: [https://huggingface.co/datasets/AdaptVis/all_datasets](https://huggingface.co/datasets/AdaptVis/all_datasets)
- Dataset loading and evaluation logic: `dataset_zoo/aro_datasets.py`
- QA prompts: `prompts/`

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

Run our margin+continuous method configuration:

```bash
python3 main_aro.py \
  --dataset Controlled_Images_A \
  --model-name llava1.5 \
  --download \
  --method adapt_vis \
  --weight1 0.5 \
  --weight2 1.5 \
  --threshold 0.64 \
  --uncertainty-criterion margin \
  --adapt-weighting sigmoid \
  --adapt-alpha 8 \
  --adapt-span 0.1 \
  --uncertainty-token-window 3 \
  --uncertainty-token-agg mean \
  --option four \
  --output-dir outputs/control_image_a_margin_continuous_100
```

To reproduce our main experiment results (Table 1 in the paper), run:

```bash
bash run.sh
```

If this doesn't work, you can check the hyperparameters in Appendix C in the paper and run manually.

## Multi-Head YOLO Overlap and AUROC

General 4-dataset analyzer:

```bash
python3 tools/analyze_head_overlap_auroc.py \
  --dataset-name Controlled_Images_A \
  --attn-dir output/Controlled_Images_A_weight1.00 \
  --result-jsonl outputs/control_image_a_margin_continuous_100/res.json \
  --sample-count 100 \
  --layer 17 \
  --yolo-model yolov8n.pt \
  --output-dir outputs/head_verify_margin_continuous/control_a_margin_sigmoid_head_analysis \
  --title-prefix Controlled_Images_A
```

Note that you may manually adjust `--attn-dir` and `--result-jsonl` if you change them when generating results.  
If you set `SAVE_ATTN_DIR` before running `main_aro.py`, then `--attn-dir` should point to that directory.

The scripts export:

- `head_overlap_bar.png`
- `head_auroc_bar.png`
- `summary.json`

## Source Code Reference

For code review and method tracing, start from:

- `main_aro.py` (entry point)
- `model_zoo/llava15.py` (model wrapper / decoding flow)
- `model_zoo/llava/modeling_llava_scal.py` (image attention intervention path)
- `model_zoo/llama/modeling_llama_add_attn.py` (attention/head-level modifications)
- `tools/analyze_head_overlap_auroc.py` (per-head overlap + AUROC)
- `tools/eval_yolo_attention_overlap.py` (YOLO overlap comparison and visual diagnostics)

