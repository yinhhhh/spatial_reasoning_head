import argparse
import os
import pandas as pd
import pdb
from model_zoo import get_model
from dataset_zoo import get_dataset
from misc import seed_all, _default_collate, save_scores
import numpy as np
import random
from torch.utils.data import DataLoader
import torch

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--model-name", default="llava1.5", type=str, \
            choices=[ "llava1.5","llava1.6"])
    parser.add_argument("--dataset", default="Controlled_Images_A", type=str, \
            choices=[ "Controlled_Images_A", "Controlled_Images_B", \
            "COCO_QA_one_obj", "COCO_QA_two_obj", "VG_QA_one_obj", "VG_QA_two_obj", "VSR"])
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--method",  type=str)
    parser.add_argument("--dola-decoding",   action="store_true")
    parser.add_argument("--info-layer",   type=int)
    parser.add_argument("--download", action="store_true", help="Whether to download the dataset if it doesn't exist. (Default: False)")
    parser.add_argument("--save-scores", action="store_true", help="Whether to save the scores for the retrieval to analyze later.")
    parser.add_argument("--output-dir", default="./outputs", type=str)
    parser.add_argument("--weight", default=1.0, type=float)
    parser.add_argument("--weight1", default=1.0, type=float)
    parser.add_argument("--weight2", default=1.0, type=float)
    parser.add_argument("--threshold", default=1.0, type=float)
    parser.add_argument("--option", default='four', type=str, choices=['two','four','six'])
    parser.add_argument(
        "--uncertainty-criterion",
        default="max_prob",
        choices=["max_prob", "margin"],
        help="Criterion used in adapt_vis to choose between weight1 and weight2.",
    )
    parser.add_argument(
        "--adapt-weighting",
        default="hard",
        choices=["hard", "linear", "sigmoid"],
        help="Weighting strategy in adapt_vis: hard switch or continuous interpolation.",
    )
    parser.add_argument(
        "--adapt-alpha",
        default=12.0,
        type=float,
        help="Slope for sigmoid weighting in adapt_vis.",
    )
    parser.add_argument(
        "--adapt-span",
        default=0.1,
        type=float,
        help="Half-width around threshold for linear weighting in adapt_vis.",
    )
    parser.add_argument(
        "--uncertainty-token-window",
        default=1,
        type=int,
        help="Number of generated token steps to aggregate for uncertainty.",
    )
    parser.add_argument(
        "--uncertainty-token-agg",
        default="mean",
        choices=["mean"],
        help="Aggregation method across token steps for uncertainty.",
    )
    parser.add_argument(
        "--adjust-method",
        default="none",
        type=str,
        help='Attention adjustment mode. Use "smooth:14-31" to apply linear smoothing only on a layer range.',
    )
    parser.add_argument(
        "--region-config",
        default="",
        type=str,
        help='Per-region adapt config, e.g. "1-13:0.9,1.05,0.35;14-25:0.3,1.35,0.3;26-31:0.8,1.15,0.32".',
    )
    parser.add_argument(
        "--low-random-th",
        default=-1.0,
        type=float,
        help="If uncertainty is below this threshold, apply near-zero random layer weights on selected ranges.",
    )
    parser.add_argument(
        "--random-mid-layers",
        default="14-25",
        type=str,
        help='Layer range for middle randomization when uncertainty < low-random-th, e.g. "14-25".',
    )
    parser.add_argument(
        "--random-late-layers",
        default="26-31",
        type=str,
        help='Layer range for late randomization when uncertainty < low-random-th, e.g. "26-31".',
    )
    parser.add_argument(
        "--random-mid-range",
        default="0.02,0.12",
        type=str,
        help='Uniform range for middle random weights when uncertainty < low-random-th, e.g. "0.02,0.12".',
    )
    parser.add_argument(
        "--random-late-range",
        default="0.01,0.08",
        type=str,
        help='Uniform range for late random weights when uncertainty < low-random-th, e.g. "0.01,0.08".',
    )
    parser.add_argument(
        "--ablate-head",
        default=-1,
        type=int,
        help="Head index to down-weight for screening. -1 disables head ablation.",
    )
    parser.add_argument(
        "--ablate-head-weight",
        default=0.05,
        type=float,
        help="Multiplicative factor applied to the ablated head on image-token attention.",
    )
    parser.add_argument(
        "--ablate-head-layers",
        default="14-31",
        type=str,
        help='Layer range where head ablation is applied, e.g. "14-31".',
    )
    parser.add_argument(
        "--active-heads",
        default="",
        type=str,
        help='Comma-separated head indices to apply scaling on, e.g. "0,2,8,9". Empty means all heads.',
    )
    parser.add_argument(
        "--enable-nonsquare-scaling",
        action="store_true",
        help="Apply attention scaling when attention map is non-square (e.g., decode with cache).",
    )
    parser.add_argument(
        "--text-attn-scale",
        default=1.0,
        type=float,
        help="Scale factor for attention to text tokens during decoding.",
    )
    parser.add_argument(
        "--image-attn-scale",
        default=1.0,
        type=float,
        help="Scale factor for attention to image tokens during decoding.",
    )
    parser.add_argument(
        "--object-text-attn-scale",
        default=1.0,
        type=float,
        help='Extra scale factor for target object text tokens (e.g., "Where is the xxx").',
    )
    parser.add_argument(
        "--top-flat-fraction",
        default=0.0,
        type=float,
        help="Optional fraction of top image-attention patches to flatten toward max during generation.",
    )
    parser.add_argument(
        "--top-flat-mix",
        default=0.0,
        type=float,
        help="Mix factor for top-flat operation: 1.0 means fully set selected values to max.",
    )
    parser.add_argument(
        "--spatial-yolo-scale",
        default=1.0,
        type=float,
        help="If >1, scale selected heads on YOLO patch region by this factor.",
    )
    parser.add_argument(
        "--spatial-yolo-conf",
        default=0.15,
        type=float,
        help="YOLO confidence threshold used to build spatial patch mask.",
    )
    parser.add_argument(
        "--spatial-yolo-box-scale",
        default=1.2,
        type=float,
        help="Scale factor for YOLO box width/height when building region mask.",
    )
    parser.add_argument(
        "--spatial-yolo-single-box-only",
        action="store_true",
        help="Only apply spatial scaling when exactly one YOLO box is detected.",
    )
    parser.add_argument(
        "--spatial-yolo-pull-ratio",
        default=0.0,
        type=float,
        help="If >0, pull YOLO-region attention up to at least (ratio * row_max) on selected heads.",
    )

    return parser.parse_args()


def main(args):
    seed_all(args.seed) 
    os.makedirs(args.output_dir, exist_ok=True)
    model, image_preprocess = get_model(args.model_name, args.device, args.method)
    dataset = get_dataset(args.dataset, image_preprocess=image_preprocess, download=args.download)
    SAMPLE=True
    TEST=os.getenv('TEST_MODE', 'False') == 'True'
    TEST_SAMPLE_COUNT = int(os.getenv('TEST_SAMPLE_COUNT', '80'))
    sampled_indices=None
    collate_fn = _default_collate if image_preprocess is None else None

    #split val and test set    
    if SAMPLE==True:  
        total_data_count = len(dataset)
        idx_file_path = f'./output/sampled_idx_{args.dataset}.npy'
        if os.path.exists(idx_file_path):
            sampled_indices = np.load(idx_file_path).tolist()
        else:
            sampled_indices = random.sample(range(total_data_count), int(0.2 * total_data_count))
            sampled_indices.sort()
            np.save(idx_file_path, np.array(sampled_indices))
        all_indices = set(range(total_data_count))
        # use test set
        if TEST==True:
            unsampled_indices = list(all_indices - set(sampled_indices))
            unsampled_indices.sort()
            sampled_indices = unsampled_indices[:min(TEST_SAMPLE_COUNT, len(unsampled_indices))]
        sub_dataset = torch.utils.data.Subset(dataset, sampled_indices)
        joint_loader = DataLoader(sub_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    #use full set
    else:       
        joint_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    print(args.dataset,args.model_name)
    if args.dataset=='VSR':
        labels=dataset.get_labels()
        scores = model.get_judge_scores_vsr_batched(
            dataset=args.dataset,
            joint_loader=joint_loader,
            method=args.method,
            weight=args.weight,
            threshold=args.threshold,
            weight1=args.weight1,
            weight2=args.weight2,
            uncertainty_criterion=args.uncertainty_criterion,
            adapt_weighting=args.adapt_weighting,
            adapt_alpha=args.adapt_alpha,
            adapt_span=args.adapt_span,
            uncertainty_token_window=args.uncertainty_token_window,
            uncertainty_token_agg=args.uncertainty_token_agg,
            adjust_method=args.adjust_method,
            region_config=args.region_config,
            low_random_th=args.low_random_th,
            random_mid_layers=args.random_mid_layers,
            random_late_layers=args.random_late_layers,
            random_mid_range=args.random_mid_range,
            random_late_range=args.random_late_range,
            ablate_head=args.ablate_head,
            ablate_head_weight=args.ablate_head_weight,
            ablate_head_layers=args.ablate_head_layers,
            active_heads=args.active_heads,
            enable_nonsquare_scaling=args.enable_nonsquare_scaling,
            text_attn_scale=args.text_attn_scale,
            image_attn_scale=args.image_attn_scale,
            object_text_attn_scale=args.object_text_attn_scale,
            top_flat_fraction=args.top_flat_fraction,
            top_flat_mix=args.top_flat_mix,
            spatial_yolo_scale=args.spatial_yolo_scale,
            spatial_yolo_conf=args.spatial_yolo_conf,
            spatial_yolo_box_scale=args.spatial_yolo_box_scale,
            spatial_yolo_single_box_only=args.spatial_yolo_single_box_only,
            spatial_yolo_pull_ratio=args.spatial_yolo_pull_ratio,
        )
        result_records = dataset.evaluate_scores(args.model_name,scores, labels, args.output_dir,args.dataset)
   

    elif args.dataset in ['Controlled_Images_B','Controlled_Images_A']:    
        scores, correct_id = model.get_out_scores_wh_batched(
            dataset=args.dataset,
            joint_loader=joint_loader,
            method=args.method,
            weight=args.weight,
            option=args.option,
            threshold=args.threshold,
            weight1=args.weight1,
            weight2=args.weight2,
            uncertainty_criterion=args.uncertainty_criterion,
            adapt_weighting=args.adapt_weighting,
            adapt_alpha=args.adapt_alpha,
            adapt_span=args.adapt_span,
            uncertainty_token_window=args.uncertainty_token_window,
            uncertainty_token_agg=args.uncertainty_token_agg,
            adjust_method=args.adjust_method,
            region_config=args.region_config,
            low_random_th=args.low_random_th,
            random_mid_layers=args.random_mid_layers,
            random_late_layers=args.random_late_layers,
            random_mid_range=args.random_mid_range,
            random_late_range=args.random_late_range,
            ablate_head=args.ablate_head,
            ablate_head_weight=args.ablate_head_weight,
            ablate_head_layers=args.ablate_head_layers,
            active_heads=args.active_heads,
            enable_nonsquare_scaling=args.enable_nonsquare_scaling,
            text_attn_scale=args.text_attn_scale,
            image_attn_scale=args.image_attn_scale,
            object_text_attn_scale=args.object_text_attn_scale,
            top_flat_fraction=args.top_flat_fraction,
            top_flat_mix=args.top_flat_mix,
            spatial_yolo_scale=args.spatial_yolo_scale,
            spatial_yolo_conf=args.spatial_yolo_conf,
            spatial_yolo_box_scale=args.spatial_yolo_box_scale,
            spatial_yolo_single_box_only=args.spatial_yolo_single_box_only,
            spatial_yolo_pull_ratio=args.spatial_yolo_pull_ratio,
        )
        print("Got the following shape of scores",scores.shape)
        # change from (82, 4, 1) to (82, 1, 4)
        scores = scores.transpose(0,2,1)
        dataset.evaluate_scores(
            scores,
            args.output_dir,
            args.dataset,
            args.model_name,
            args.method,
            args.weight,
            sampled_indices,
            args.option,
            args.weight1,
            args.weight2,
            args.threshold,
            args.uncertainty_criterion,
        )
        # dataset.save_scores(scores,correct_id,args.output_dir,args.dataset,args.method,args.weight,args.model_name,args.option)

    else:
        scores,correct_id = model.get_out_scores_wh_batched(
            dataset=args.dataset,
            joint_loader=joint_loader,
            method=args.method,
            weight=args.weight,
            option=args.option,
            threshold=args.threshold,
            weight1=args.weight1,
            weight2=args.weight2,
            uncertainty_criterion=args.uncertainty_criterion,
            adapt_weighting=args.adapt_weighting,
            adapt_alpha=args.adapt_alpha,
            adapt_span=args.adapt_span,
            uncertainty_token_window=args.uncertainty_token_window,
            uncertainty_token_agg=args.uncertainty_token_agg,
            adjust_method=args.adjust_method,
            region_config=args.region_config,
            low_random_th=args.low_random_th,
            random_mid_layers=args.random_mid_layers,
            random_late_layers=args.random_late_layers,
            random_mid_range=args.random_mid_range,
            random_late_range=args.random_late_range,
            ablate_head=args.ablate_head,
            ablate_head_weight=args.ablate_head_weight,
            ablate_head_layers=args.ablate_head_layers,
            active_heads=args.active_heads,
            enable_nonsquare_scaling=args.enable_nonsquare_scaling,
            text_attn_scale=args.text_attn_scale,
            image_attn_scale=args.image_attn_scale,
            object_text_attn_scale=args.object_text_attn_scale,
            top_flat_fraction=args.top_flat_fraction,
            top_flat_mix=args.top_flat_mix,
            spatial_yolo_scale=args.spatial_yolo_scale,
            spatial_yolo_conf=args.spatial_yolo_conf,
            spatial_yolo_box_scale=args.spatial_yolo_box_scale,
            spatial_yolo_single_box_only=args.spatial_yolo_single_box_only,
            spatial_yolo_pull_ratio=args.spatial_yolo_pull_ratio,
        )
        dataset.save_scores(scores,correct_id,args.output_dir,args.dataset,args.method,args.weight,args.model_name,args.option)

        
if __name__ == "__main__":
    args = config()
    main(args)
