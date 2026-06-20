import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
# from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import random
from transformers import AutoProcessor, LlamaTokenizerFast, CLIPImageProcessor
import pdb
# import probe_llava
from .llava import  LlavaForConditionalGeneration, LlavaForConditionalGenerationScal

import torch
import torch.nn.functional as F
from PIL import Image
import requests
import json
import os
from collections import Counter
import re
# from model_zoo.utils import normalize_answer,chat_completion_request,run_conversation

from PIL import Image
import math
MODEL='llava-hf/llava-1.5-7b-hf'

import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import transformers
from transformers.generation.utils import SampleOutput, SampleDecoderOnlyOutput, SampleEncoderDecoderOutput,GenerateEncoderDecoderOutput,GenerateDecoderOnlyOutput,GenerateNonBeamOutput
import os
import json
import random
import numpy as np
import torch
from tqdm import tqdm


def _compute_uncertainty(scores_steps, criterion="max_prob", token_window=1, token_agg="mean"):
    if isinstance(scores_steps, torch.Tensor):
        step_tensors = [scores_steps]
    else:
        step_tensors = list(scores_steps)
    if len(step_tensors) == 0:
        return 0.0

    use_steps = step_tensors[: max(1, min(token_window, len(step_tensors)))]
    vals = []
    for step_scores in use_steps:
        probs = torch.nn.functional.softmax(step_scores, dim=-1)[0]
        top2_vals, _ = torch.topk(probs, 2)
        if criterion == "margin":
            vals.append(float(top2_vals[0] - top2_vals[1]))
        else:
            vals.append(float(top2_vals[0]))

    if token_agg == "mean":
        return np.round(float(np.mean(vals)), 4)
    return np.round(float(np.mean(vals)), 4)


def _compute_top2_margin_stats(scores_steps, token_window=1, token_agg="mean"):
    if isinstance(scores_steps, torch.Tensor):
        step_tensors = [scores_steps]
    else:
        step_tensors = list(scores_steps)
    if len(step_tensors) == 0:
        return {"top1": 0.0, "top2": 0.0, "margin": 0.0}

    use_steps = step_tensors[: max(1, min(token_window, len(step_tensors)))]
    top1_vals = []
    top2_vals = []
    margin_vals = []
    for step_scores in use_steps:
        probs = torch.nn.functional.softmax(step_scores, dim=-1)[0]
        top2, _ = torch.topk(probs, 2)
        t1 = float(top2[0])
        t2 = float(top2[1])
        top1_vals.append(t1)
        top2_vals.append(t2)
        margin_vals.append(t1 - t2)

    # Only mean is currently used, keep behavior explicit.
    if token_agg == "mean":
        top1 = float(np.mean(top1_vals))
        top2 = float(np.mean(top2_vals))
        margin = float(np.mean(margin_vals))
    else:
        top1 = float(np.mean(top1_vals))
        top2 = float(np.mean(top2_vals))
        margin = float(np.mean(margin_vals))

    return {
        "top1": np.round(top1, 4),
        "top2": np.round(top2, 4),
        "margin": np.round(margin, 4),
    }


def _continuous_weight(
    uncertainty,
    threshold,
    weight1,
    weight2,
    mode="hard",
    alpha=12.0,
    span=0.1,
):
    if mode == "hard":
        return weight1 if uncertainty < threshold else weight2

    if mode == "linear":
        lo = threshold - max(span, 1e-6)
        hi = threshold + max(span, 1e-6)
        if uncertainty <= lo:
            t = 0.0
        elif uncertainty >= hi:
            t = 1.0
        else:
            t = (uncertainty - lo) / (hi - lo)
        return weight1 + (weight2 - weight1) * t

    # sigmoid
    t = 1.0 / (1.0 + np.exp(-alpha * (uncertainty - threshold)))
    return weight1 + (weight2 - weight1) * float(t)


def _parse_region_config(region_config: str):
    """
    Parse per-region config string:
    "1-13:w1,w2,th;14-25:w1,w2,th;26-31:w1,w2,th"
    """
    if not region_config or not region_config.strip():
        return []

    parsed = []
    chunks = [chunk.strip() for chunk in region_config.split(";") if chunk.strip()]
    for chunk in chunks:
        if ":" not in chunk:
            continue
        layer_span, vals = chunk.split(":", 1)
        if "-" not in layer_span:
            continue
        start_str, end_str = layer_span.split("-", 1)
        triplet = [v.strip() for v in vals.split(",")]
        if len(triplet) != 3:
            continue
        try:
            start = int(start_str)
            end = int(end_str)
            w1 = float(triplet[0])
            w2 = float(triplet[1])
            th = float(triplet[2])
        except ValueError:
            continue
        if start > end:
            start, end = end, start
        parsed.append({"start": start, "end": end, "w1": w1, "w2": w2, "th": th})
    return parsed


def _parse_layer_span(span_text: str):
    try:
        left, right = span_text.split("-", 1)
        a, b = int(left.strip()), int(right.strip())
    except (ValueError, AttributeError):
        return None
    if a > b:
        a, b = b, a
    return a, b


def _parse_random_range(range_text: str, fallback_low: float, fallback_high: float):
    try:
        low_s, high_s = range_text.split(",", 1)
        low, high = float(low_s.strip()), float(high_s.strip())
    except (ValueError, AttributeError):
        low, high = fallback_low, fallback_high
    if low > high:
        low, high = high, low
    return low, high


def _build_region_layer_weights(
    region_config: str,
    uncertainty: float,
    layer_count: int = 32,
    low_random_th: float = -1.0,
    random_mid_layers: str = "14-25",
    random_late_layers: str = "26-31",
    random_mid_range: str = "0.02,0.12",
    random_late_range: str = "0.01,0.08",
):
    segments = _parse_region_config(region_config)
    if not segments:
        return None
    layer_weights = [1.0] * layer_count
    for seg in segments:
        w = seg["w1"] if uncertainty < seg["th"] else seg["w2"]
        left = max(0, seg["start"])
        right = min(layer_count - 1, seg["end"])
        for i in range(left, right + 1):
            layer_weights[i] = float(w)

    if low_random_th >= 0.0 and uncertainty < low_random_th:
        mid_span = _parse_layer_span(random_mid_layers)
        late_span = _parse_layer_span(random_late_layers)
        mid_lo, mid_hi = _parse_random_range(random_mid_range, 0.02, 0.12)
        late_lo, late_hi = _parse_random_range(random_late_range, 0.01, 0.08)

        if mid_span is not None:
            left = max(0, mid_span[0])
            right = min(layer_count - 1, mid_span[1])
            for i in range(left, right + 1):
                layer_weights[i] = float(np.random.uniform(mid_lo, mid_hi))

        if late_span is not None:
            left = max(0, late_span[0])
            right = min(layer_count - 1, late_span[1])
            for i in range(left, right + 1):
                layer_weights[i] = float(np.random.uniform(late_lo, late_hi))
    return layer_weights


def _build_head_weights(num_heads: int, ablate_head: int = -1, ablate_weight: float = 0.05):
    if ablate_head < 0 or ablate_head >= num_heads:
        return None
    vals = [1.0] * num_heads
    vals[int(ablate_head)] = float(ablate_weight)
    return vals


def _parse_active_heads(active_heads: str, num_heads: int = 32):
    if not active_heads or not active_heads.strip():
        return None
    mask = [0.0] * num_heads
    for token in active_heads.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            idx = int(token)
        except ValueError:
            continue
        if 0 <= idx < num_heads:
            mask[idx] = 1.0
    if sum(mask) == 0:
        return None
    return mask


_YOLO_MODEL_CACHE = {}


def _get_yolo_model(model_path: str = "yolov8n.pt"):
    if model_path not in _YOLO_MODEL_CACHE:
        from ultralytics import YOLO

        _YOLO_MODEL_CACHE[model_path] = YOLO(model_path)
    return _YOLO_MODEL_CACHE[model_path]


def _build_yolo_patch_mask_from_image(
    image_pil,
    conf: float = 0.15,
    box_scale: float = 1.2,
    require_single_box: bool = True,
    patch_side: int = 24,
):
    model = _get_yolo_model("yolov8n.pt")
    det = model.predict(source=image_pil, conf=conf, verbose=False)[0]
    if det.boxes is None or len(det.boxes) == 0:
        return None
    if require_single_box and len(det.boxes) != 1:
        return None
    if len(det.boxes) > 1:
        confs = det.boxes.conf.detach().cpu().numpy()
        keep = int(np.argmax(confs))
        box = det.boxes.xyxy.detach().cpu().numpy()[keep]
    else:
        box = det.boxes.xyxy.detach().cpu().numpy()[0]

    img_w, img_h = image_pil.size
    x1, y1, x2, y2 = box.tolist()
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = max(1e-6, x2 - x1) * float(box_scale)
    bh = max(1e-6, y2 - y1) * float(box_scale)
    sx1 = max(0.0, cx - 0.5 * bw)
    sy1 = max(0.0, cy - 0.5 * bh)
    sx2 = min(float(img_w - 1), cx + 0.5 * bw)
    sy2 = min(float(img_h - 1), cy + 0.5 * bh)

    x_centers = (np.arange(patch_side) + 0.5) * (img_w / patch_side)
    y_centers = (np.arange(patch_side) + 0.5) * (img_h / patch_side)
    xv, yv = np.meshgrid(x_centers, y_centers)
    mask = ((xv >= sx1) & (xv <= sx2) & (yv >= sy1) & (yv <= sy2)).astype(np.int32).reshape(-1)
    if mask.sum() <= 0:
        return None
    return mask.tolist()


def _extract_where_is_object(prompt: str) -> Optional[str]:
    if not prompt:
        return None
    m = re.search(r"where\s+is\s+the\s+(.+?)\s+in\s+the\s+photo", prompt, flags=re.IGNORECASE)
    if not m:
        return None
    phrase = m.group(1).strip(" .,:;!?")
    return phrase if phrase else None


def _find_subsequence_spans(sequence: List[int], pattern: List[int]) -> List[Tuple[int, int]]:
    if not pattern or len(pattern) > len(sequence):
        return []
    spans = []
    p_len = len(pattern)
    for start in range(len(sequence) - p_len + 1):
        if sequence[start : start + p_len] == pattern:
            spans.append((start, start + p_len))
    return spans


def _build_object_token_mask(input_ids: torch.Tensor, tokenizer, prompt: str):
    if tokenizer is None:
        return None
    phrase = _extract_where_is_object(prompt)
    if phrase is None:
        return None

    seq = input_ids[0].tolist()
    candidates = []
    for text in [phrase, " " + phrase]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            candidates.append(ids)
    if not candidates:
        return None

    spans: List[Tuple[int, int]] = []
    for cand in candidates:
        spans.extend(_find_subsequence_spans(seq, cand))
    if not spans:
        return None

    merged = [torch.zeros_like(row, dtype=torch.long) for row in input_ids]
    for start, end in spans:
        merged[0][start:end] = 1
    return merged


def _add_weight_greedy_search(
    self,
    input_ids: torch. LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    output_logits: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    # keys:Optional[torch.Tensor] = None,
    weight: Optional[float] = None,
    adjust_method: Optional[str] = None,
    layer_weights: Optional[List[float]] = None,
    head_weights: Optional[List[float]] = None,
    head_apply_layers: Optional[str] = None,
    active_head_mask: Optional[List[float]] = None,
    enable_nonsquare_scaling: bool = False,
    text_attn_scale: float = 1.0,
    image_attn_scale: float = 1.0,
    object_attn_scale: float = 1.0,
    top_flat_fraction: float = 0.0,
    top_flat_mix: float = 0.0,
    pos: Optional[torch.Tensor] = None,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    scores = () if (return_dict_in_generate and output_scores) else None
    before = () if (return_dict_in_generate) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    if "inputs_embeds" in model_kwargs:
        cur_len = model_kwargs["inputs_embeds"].shape[1]
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)
    
    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        import pdb
        # 
        if 'Scal' not in str(type(self)):
            outputs = self(
                **model_inputs,
               
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        else:
            
            outputs = self(
                **model_inputs,
                weight=weight,
                adjust_method=adjust_method,
                layer_weights=layer_weights,
                head_weights=head_weights,
                head_apply_layers=head_apply_layers,
                active_head_mask=active_head_mask,
                enable_nonsquare_scaling=enable_nonsquare_scaling,
                text_attn_scale=text_attn_scale,
                image_attn_scale=image_attn_scale,
                object_attn_scale=object_attn_scale,
                top_flat_fraction=top_flat_fraction,
                top_flat_mix=top_flat_mix,
                pos=pos,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_tokens_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_tokens_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # argmax
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids
    
def change_greedy_to_add_weight():
    transformers.generation.utils.GenerationMixin._greedy_search = _add_weight_greedy_search

class LlavaWrapper:
    def __init__(self, root_dir, device,method):
        
        if method=='scaling_vis' or method=='adapt_vis':
            self.model = LlavaForConditionalGenerationScal.from_pretrained(MODEL, revision='a272c74',cache_dir=root_dir,ignore_mismatched_sizes=True).eval().to(device)

        else:
            self.model = LlavaForConditionalGeneration.from_pretrained(MODEL, revision='a272c74', cache_dir=root_dir,ignore_mismatched_sizes=True).eval().to(device)

        self.feature_extractor = CLIPImageProcessor.from_pretrained(MODEL, revision='a272c74',cache_dir=root_dir)
        self.tokenizer = LlamaTokenizerFast.from_pretrained(MODEL, revision='a272c74',cache_dir=root_dir)
        self.processor = AutoProcessor.from_pretrained(MODEL, revision='a272c74',cache_dir=root_dir)

        self.device = device
    
    @torch.no_grad()
    def get_text_embeddings(self, texts, text_batch_size=64, normalize=False):
        num_text = len(texts)
        text_embeds = []
        for i in tqdm(range(0, num_text, text_batch_size)):
            text = texts[i: min(num_text, i+text_batch_size)]
            text_input = self.tokenizer(text=text, return_tensors="pt", padding="max_length", max_length=77).to(self.device)
            text_feats = self.model.llava.get_text_features(**text_input).cpu().numpy()[:, 0, :].to(self.device)
            if normalize:
                text_feats = text_feats / np.linalg.norm(text_feats, axis=1, keepdims=True)          
            text_embeds.append(text_feats)   
            
        return np.concatenate(text_embeds, axis=0)
    
    @torch.no_grad()
    def get_image_embeddings(self, image_loader, normalize=False):
        image_embeds = []
        for batch in tqdm(image_loader):
            images = batch["image"]
            inputs = self.feature_extractor(images=images, return_tensors="pt").to(self.device)
            image_feats = self.model.llava.get_image_features(**inputs).cpu().numpy()[:, 0, :]
            if normalize:
                image_feats = image_feats / np.linalg.norm(image_feats, axis=1, keepdims=True)
            image_embeds.append(image_feats)

        return np.concatenate(image_embeds, axis=0)
    
    
    def get_retrieval_scores_dataset(self, loader):
        texts = loader.dataset.text
        text_embeds = self.get_text_embeddings(texts, normalize=True)
        image_embeds = self.get_image_embeddings(loader, normalize=True)
        scores = image_embeds @ text_embeds.T
        return scores
    
    
    @torch.no_grad()
    def get_out_scores_wh_batched(
        self,
        dataset,
        joint_loader,
        method,
        weight,
        option,
        threshold,
        weight1,
        weight2,
        uncertainty_criterion="max_prob",
        adapt_weighting="hard",
        adapt_alpha=12.0,
        adapt_span=0.1,
        uncertainty_token_window=1,
        uncertainty_token_agg="mean",
        adjust_method="none",
        region_config="",
        low_random_th=-1.0,
        random_mid_layers="14-25",
        random_late_layers="26-31",
        random_mid_range="0.02,0.12",
        random_late_range="0.01,0.08",
        ablate_head=-1,
        ablate_head_weight=0.05,
        ablate_head_layers="14-31",
        active_heads="",
        enable_nonsquare_scaling=False,
        text_attn_scale=1.0,
        image_attn_scale=1.0,
        object_text_attn_scale=1.0,
        top_flat_fraction=0.0,
        top_flat_mix=0.0,
        spatial_yolo_scale=1.0,
        spatial_yolo_conf=0.15,
        spatial_yolo_box_scale=1.2,
        spatial_yolo_single_box_only=False,
        spatial_yolo_pull_ratio=0.0,
    ):

        
        scores = []  # To store scores for each batch
        index_of_total = 0  # Track total number of prompts processed
        acc = 0  # Track the number of correct predictions
        correct_id = []  # Track indices of correct predictions

        # Determine the correct question-answer file based on the dataset
        qst_ans_file = f'prompts/{dataset}_with_answer_{option}_options.jsonl'
        
        # Load prompts and answers from the question-answer file
        with open(qst_ans_file, 'r') as file:
            prompt_list = []
            answer_list = []
            first_prompt_list = []
            second_prompt_list = []
            for line in file:
                data = json.loads(line)
                # Select prompt based on mode
                
                prompt_list.append(data["question"])
                
                # Store additional prompts if adjustment method is 'sub'
                
                answer_list.append(data["answer"])

        # Sampling configuration
        SAMPLE = True
        TEST = os.getenv('TEST_MODE', 'False') == 'True'
        TEST_SAMPLE_COUNT = int(os.getenv('TEST_SAMPLE_COUNT', '80'))
        total_data_count = len(prompt_list)
        
        # Perform sampling if enabled
        if SAMPLE:
            idx_file_path = f'./output/sampled_idx_{dataset}.npy'
            
            if os.path.exists(idx_file_path):
                sampled_indices = np.load(idx_file_path).tolist()
            else:
                sampled_indices = random.sample(range(total_data_count), int(0.2 * total_data_count))
                sampled_indices.sort()
                np.save(idx_file_path, np.array(sampled_indices))

            # For testing mode, use unsampled indices
            if TEST:
                all_indices = set(range(total_data_count))
                unsampled_indices = list(all_indices - set(sampled_indices))
                unsampled_indices.sort()
                sampled_indices = unsampled_indices[:min(TEST_SAMPLE_COUNT, len(unsampled_indices))]

            # Subset prompts and answers based on sampled indices
            prompt_list = [prompt_list[i] for i in sampled_indices]
            answer_list = [answer_list[i] for i in sampled_indices]

        # Create directory for saving attention maps
        save_attn_dir = f"./output/{dataset}_weight{weight:.2f}"
        os.makedirs(save_attn_dir, exist_ok=True)
        active_head_mask = _parse_active_heads(active_heads, num_heads=32)

        results = []  # Store results for each generated sequence
        for batch in tqdm(joint_loader):
            batch_scores = []
            
            # Set environment variable for attention map save path
            os.environ['SAVE_ATTN_PATH'] = f'{save_attn_dir}/{index_of_total}/'
            os.makedirs(os.environ['SAVE_ATTN_PATH'], exist_ok=True)

            # Iterate over each image option in the batch
            for i_option in batch["image_options"]:
                im_scores = []
                
                for _ in i_option:
                    prompt = prompt_list[index_of_total]
                    yolo_patch_mask = None
                    if float(spatial_yolo_scale) > 1.0 or float(spatial_yolo_pull_ratio) > 0.0:
                        try:
                            yolo_patch_mask = _build_yolo_patch_mask_from_image(
                                image_pil=_,
                                conf=float(spatial_yolo_conf),
                                box_scale=float(spatial_yolo_box_scale),
                                require_single_box=bool(spatial_yolo_single_box_only),
                                patch_side=24,
                            )
                        except Exception:
                            yolo_patch_mask = None
                    if yolo_patch_mask is not None:
                        os.environ["YOLO_PATCH_MASK"] = ",".join(str(int(v)) for v in yolo_patch_mask)
                    else:
                        os.environ["YOLO_PATCH_MASK"] = ""
                    os.environ["YOLO_PULL_RATIO"] = str(float(spatial_yolo_pull_ratio))
                    
                    # Preprocess input for the model
                    single_input = self.processor(
                        text=prompt, images=_, padding="max_length", return_tensors="pt", max_length=77
                    ).to(self.device)
                    
                    # Create key mask for special token
                    keys = [torch.where(input_id == 32001, 1, 0) for input_id in single_input['input_ids']]
                    object_keys = _build_object_token_mask(
                        single_input["input_ids"],
                        getattr(self.processor, "tokenizer", None),
                        prompt,
                    )

                    # Generate predictions based on specified method
                    head_weights = _build_head_weights(32, ablate_head, ablate_head_weight)
                    if method == 'scaling_vis':
                        
                        change_greedy_to_add_weight()
                        output = self.model.generate(
                            **single_input, keys=keys, weight=weight,
                            adjust_method=adjust_method,
                            layer_weights=None,
                            head_weights=head_weights,
                            head_apply_layers=ablate_head_layers,
                            active_head_mask=active_head_mask,
                            enable_nonsquare_scaling=enable_nonsquare_scaling,
                            text_attn_scale=text_attn_scale,
                            image_attn_scale=image_attn_scale,
                            pos=object_keys,
                            object_attn_scale=object_text_attn_scale,
                            top_flat_fraction=top_flat_fraction,
                            top_flat_mix=top_flat_mix,
                            max_new_tokens=100, output_scores=True, return_dict_in_generate=True
                        )
                        uncertainty = _compute_uncertainty(
                            output['scores'],
                            "max_prob",
                            token_window=uncertainty_token_window,
                            token_agg=uncertainty_token_agg,
                        )
                        gen = self.processor.decode(output['sequences'][0][len(single_input['input_ids'][-1]):], skip_special_tokens=True)
                    
                    elif method == 'adapt_vis':
                        change_greedy_to_add_weight()
                       
                        output = self.model.generate(
                            **single_input, weight=1.0, adjust_method=adjust_method,
                            head_weights=head_weights,
                            head_apply_layers=ablate_head_layers,
                            active_head_mask=active_head_mask,
                            enable_nonsquare_scaling=enable_nonsquare_scaling,
                            text_attn_scale=text_attn_scale,
                            image_attn_scale=image_attn_scale,
                            pos=object_keys,
                            object_attn_scale=object_text_attn_scale,
                            top_flat_fraction=top_flat_fraction,
                            top_flat_mix=top_flat_mix,
                            max_new_tokens=100, output_scores=True, return_dict_in_generate=True
                        )
                        uncertainty = _compute_uncertainty(
                            output['scores'],
                            uncertainty_criterion,
                            token_window=uncertainty_token_window,
                            token_agg=uncertainty_token_agg,
                        )
                        if os.getenv("LOG_TOP2_MARGIN", "0") == "1":
                            _s = _compute_top2_margin_stats(
                                output['scores'],
                                token_window=uncertainty_token_window,
                                token_agg=uncertainty_token_agg,
                            )
                            print(
                                f"[top2_debug] top1={_s['top1']:.4f} top2={_s['top2']:.4f} margin={_s['margin']:.4f}"
                            )
                        # Adjust attention based on uncertainty
                        adapt_weight = _continuous_weight(
                            uncertainty,
                            threshold,
                            weight1,
                            weight2,
                            mode=adapt_weighting,
                            alpha=adapt_alpha,
                            span=adapt_span,
                        )
                        if os.getenv("LOG_ADAPT_WEIGHT", "0") == "1":
                            print(
                                f"[adapt_debug] uncertainty={uncertainty:.4f} "
                                f"threshold={threshold:.4f} adapt_weight={adapt_weight:.6f}"
                            )
                        region_layer_weights = _build_region_layer_weights(
                            region_config=region_config,
                            uncertainty=uncertainty,
                            layer_count=32,
                            low_random_th=low_random_th,
                            random_mid_layers=random_mid_layers,
                            random_late_layers=random_late_layers,
                            random_mid_range=random_mid_range,
                            random_late_range=random_late_range,
                        )
                        spatial_scale = float(spatial_yolo_scale) if float(spatial_yolo_scale) > 1.0 else 1.0
                        if spatial_scale > 1.0 and yolo_patch_mask is None:
                            spatial_scale = 1.0
                        output = self.model.generate(
                            **single_input, keys=keys, weight=adapt_weight * spatial_scale,
                            adjust_method=adjust_method,
                            layer_weights=region_layer_weights,
                            head_weights=head_weights,
                            head_apply_layers=ablate_head_layers,
                            active_head_mask=active_head_mask,
                            enable_nonsquare_scaling=enable_nonsquare_scaling,
                            text_attn_scale=text_attn_scale,
                            image_attn_scale=image_attn_scale,
                            pos=object_keys,
                            object_attn_scale=object_text_attn_scale,
                            top_flat_fraction=top_flat_fraction,
                            top_flat_mix=top_flat_mix,
                            max_new_tokens=100, output_scores=True, return_dict_in_generate=True
                        )
                        gen = self.processor.decode(output['sequences'][0][len(single_input['input_ids'][-1]):], skip_special_tokens=True)

                    else:
                        # Default generation method
                        supports_custom_attention = "Scal" in str(type(self.model))
                        if supports_custom_attention and (text_attn_scale != 1.0 or image_attn_scale != 1.0):
                            change_greedy_to_add_weight()
                            output = self.model.generate(
                                **single_input,
                                keys=keys,
                                weight=1.0,
                                adjust_method="none",
                                layer_weights=None,
                                head_weights=head_weights,
                                head_apply_layers=ablate_head_layers,
                                active_head_mask=active_head_mask,
                                enable_nonsquare_scaling=enable_nonsquare_scaling,
                                text_attn_scale=text_attn_scale,
                                image_attn_scale=image_attn_scale,
                                pos=object_keys,
                                object_attn_scale=object_text_attn_scale,
                                top_flat_fraction=top_flat_fraction,
                                top_flat_mix=top_flat_mix,
                                max_new_tokens=100,
                                output_scores=True,
                                return_dict_in_generate=True,
                            )
                        elif text_attn_scale != 1.0 or image_attn_scale != 1.0:
                            print(
                                "[warn] text/image attention scaling requested but current model "
                                "does not support custom attention kwargs; falling back to default baseline."
                            )
                            output = self.model.generate(
                                **single_input,
                                max_new_tokens=100,
                                output_scores=True,
                                return_dict_in_generate=True,
                            )
                        else:
                            output = self.model.generate(
                                **single_input,
                                max_new_tokens=100,
                                output_scores=True,
                                return_dict_in_generate=True,
                            )
                        gen = self.processor.decode(output['sequences'][0][len(single_input['input_ids'][-1]):], skip_special_tokens=True)
                        uncertainty = _compute_uncertainty(
                            output['scores'],
                            "max_prob",
                            token_window=uncertainty_token_window,
                            token_agg=uncertainty_token_agg,
                        )

                    # Print prompt, generated text, and expected answer
                    print(f"Prompt: {prompt}\nGeneration: {gen}\nGolden: {answer_list[index_of_total][0]}")
                    
                    result = {
                        "Prompt": prompt,
                        "Generation": gen,
                        "Golden": answer_list[index_of_total][0],
                    }
                    results.append(result)
                    
                    # Check if the generation matches the expected answer
                    c_option = batch["caption_options"]
                    if len(list(c_option)) == 4:
                        if (answer_list[index_of_total][0] in gen or answer_list[index_of_total][0].lower() in gen.lower()) \
                                and not (answer_list[index_of_total][0].lower() == 'on' and 'front' in gen.strip().lower()):
                            acc += 1
                            correct_id.append(index_of_total)
                            answers = [1, 0, 0, 0]
                        else:
                            answers = [0, 0, 1, 0]
                    
                    elif len(list(c_option)) == 2:
                        if (answer_list[index_of_total][0] in gen or answer_list[index_of_total][0].lower() in gen.lower()) \
                                and not (answer_list[index_of_total][0].lower() == 'on' and 'front' in gen.strip().lower()):
                            acc += 1
                            correct_id.append(index_of_total)
                            answers = [1, 0]
                        else:
                            answers = [0, 1]

                    im_scores.append(np.expand_dims(np.array(answers), -1))
                    index_of_total += 1

                batch_scores.append(np.concatenate(im_scores, axis=-1))

            scores.append(batch_scores)

            # Save results to file
            output_file_path = f'./output/results1.5_{dataset}_{method}_{weight}_{option}option_{TEST}.json'
            print("Saving results to", output_file_path)
            with open(output_file_path, 'w', encoding='utf-8') as fout:
                json.dump(results, fout, ensure_ascii=False, indent=4)
            print(acc, index_of_total, acc / index_of_total)

        # Save accuracy and correct IDs to file
        print(acc / index_of_total)
        output_score_file = output_file_path.replace(".json", "scores.json")
        with open(output_score_file, 'w', encoding='utf-8') as fout:
            json.dump({"acc": acc / index_of_total, "correct_id": correct_id}, fout, ensure_ascii=False, indent=4)

        # Concatenate all scores and return based on dataset type
        all_scores = np.concatenate(scores, axis=0)  # N x K x L
        if dataset in ['Controlled_Images_B', 'Controlled_Images_A']:
            return (all_scores, [])
        else:
            return (acc / index_of_total, correct_id)

    
    
    @torch.no_grad()
    def get_judge_scores_vsr_batched(
        self,
        dataset,
        joint_loader,
        method,
        weight,
        threshold,
        weight1,
        weight2,
        uncertainty_criterion="max_prob",
        adapt_weighting="hard",
        adapt_alpha=12.0,
        adapt_span=0.1,
        uncertainty_token_window=1,
        uncertainty_token_agg="mean",
        adjust_method="none",
        region_config="",
        low_random_th=-1.0,
        random_mid_layers="14-25",
        random_late_layers="26-31",
        random_mid_range="0.02,0.12",
        random_late_range="0.01,0.08",
        ablate_head=-1,
        ablate_head_weight=0.05,
        ablate_head_layers="14-31",
        active_heads="",
        enable_nonsquare_scaling=False,
        text_attn_scale=1.0,
        image_attn_scale=1.0,
        object_text_attn_scale=1.0,
        top_flat_fraction=0.0,
        top_flat_mix=0.0,
    ):
        
        
        index = 0
        TP, TN, FP, FN = 0, 0, 0, 0

        # Set the directory to save attention maps
        save_attn_dir = f"/home/user/shiqi/mmlm_mech/whatsup_vlms/outputs/{dataset}_weight{weight:.2f}"
        if not os.path.exists(save_attn_dir):
            print("Creating directory for saving attention maps:", save_attn_dir)
            os.makedirs(save_attn_dir)
        
        index_of_total = 0
        results = []
        active_head_mask = _parse_active_heads(active_heads, num_heads=32)

        # Process each batch in the joint loader
        for batch in tqdm(joint_loader):
            batch_scores = []
            
            # Create directory for saving attention maps for each batch
            os.environ['SAVE_ATTN_PATH'] = f'{save_attn_dir}/{index_of_total}/'
            os.makedirs(os.environ['SAVE_ATTN_PATH'], exist_ok=True)

            # Iterate over image options in the batch
            for i_option in batch["image_options"]:
                im_scores = []

                # Iterate over caption options
                for c_option in batch["caption_options"]:
                    prompt = "User: <image>\n Determine whether the description about the spatial relationship is correct or not. Answer with yes or no: "
                    qst = [prompt] * len(list(c_option))
                    end_fix = [" Assistant:"] * len(list(c_option))
                    concatenated_list = [s1 + s2 + s3 for s1, s2, s3 in zip(qst, c_option, end_fix)]
                    
                    # Generate responses for each concatenated input
                    for idx, text in enumerate(concatenated_list):
                        # Prepare input data for the model
                        single_input = self.processor(text=text, images=list(i_option)[idx], padding="max_length", return_tensors="pt", max_length=77).to(self.device)
                        keys = [torch.where(input_id == 32001, 1, 0) for input_id in single_input['input_ids']]
                        object_keys = _build_object_token_mask(
                            single_input["input_ids"],
                            getattr(self.processor, "tokenizer", None),
                            text,
                        )
                        
                        # Apply different attention adjustment methods based on the 'method' argument
                        head_weights = _build_head_weights(32, ablate_head, ablate_head_weight)
                        if method == 'scaling_vis':
                            change_greedy_to_add_weight()
                            output = self.model.generate(
                                **single_input,
                                keys=keys,
                                weight=weight,
                                adjust_method=adjust_method,
                                layer_weights=None,
                                head_weights=head_weights,
                                head_apply_layers=ablate_head_layers,
                                active_head_mask=active_head_mask,
                                enable_nonsquare_scaling=enable_nonsquare_scaling,
                                text_attn_scale=text_attn_scale,
                                image_attn_scale=image_attn_scale,
                                pos=object_keys,
                                object_attn_scale=object_text_attn_scale,
                                top_flat_fraction=top_flat_fraction,
                                top_flat_mix=top_flat_mix,
                                max_new_tokens=100,
                                output_scores=True,
                                return_dict_in_generate=True,
                            )
                            uncertainty = _compute_uncertainty(
                                output['scores'],
                                "max_prob",
                                token_window=uncertainty_token_window,
                                token_agg=uncertainty_token_agg,
                            )
                            gen = self.processor.decode(output[0][0][len(single_input['input_ids'][-1]):], skip_special_tokens=True, output_attentions=True)
                        
                        elif method == 'adapt_vis':
                            change_greedy_to_add_weight()
                            # Basic generation step
                            output = self.model.generate(
                                **single_input,
                                weight=1.0,
                                adjust_method=adjust_method,
                                head_weights=head_weights,
                                head_apply_layers=ablate_head_layers,
                                active_head_mask=active_head_mask,
                                enable_nonsquare_scaling=enable_nonsquare_scaling,
                                text_attn_scale=text_attn_scale,
                                image_attn_scale=image_attn_scale,
                                pos=object_keys,
                                object_attn_scale=object_text_attn_scale,
                                top_flat_fraction=top_flat_fraction,
                                top_flat_mix=top_flat_mix,
                                max_new_tokens=100,
                                output_scores=True,
                                return_dict_in_generate=True,
                            )
                            gen = self.processor.decode(output['sequences'][0][len(single_input['input_ids'][-1]):], skip_special_tokens=True, output_attentions=True)
                            uncertainty = _compute_uncertainty(
                                output['scores'],
                                uncertainty_criterion,
                                token_window=uncertainty_token_window,
                                token_agg=uncertainty_token_agg,
                            )
                            if os.getenv("LOG_TOP2_MARGIN", "0") == "1":
                                _s = _compute_top2_margin_stats(
                                    output['scores'],
                                    token_window=uncertainty_token_window,
                                    token_agg=uncertainty_token_agg,
                                )
                                print(
                                    f"[top2_debug] top1={_s['top1']:.4f} top2={_s['top2']:.4f} margin={_s['margin']:.4f}"
                                )
                            
                            # Apply weighted generation based on uncertainty
                            adapt_weight = _continuous_weight(
                                uncertainty,
                                threshold,
                                weight1,
                                weight2,
                                mode=adapt_weighting,
                                alpha=adapt_alpha,
                                span=adapt_span,
                            )
                            region_layer_weights = _build_region_layer_weights(
                                region_config=region_config,
                                uncertainty=uncertainty,
                                layer_count=32,
                                low_random_th=low_random_th,
                                random_mid_layers=random_mid_layers,
                                random_late_layers=random_late_layers,
                                random_mid_range=random_mid_range,
                                random_late_range=random_late_range,
                            )
                            output = self.model.generate(
                                **single_input,
                                keys=keys,
                                weight=adapt_weight,
                                adjust_method=adjust_method,
                                layer_weights=region_layer_weights,
                                head_weights=head_weights,
                                head_apply_layers=ablate_head_layers,
                                active_head_mask=active_head_mask,
                                enable_nonsquare_scaling=enable_nonsquare_scaling,
                                text_attn_scale=text_attn_scale,
                                image_attn_scale=image_attn_scale,
                                pos=object_keys,
                                object_attn_scale=object_text_attn_scale,
                                top_flat_fraction=top_flat_fraction,
                                top_flat_mix=top_flat_mix,
                                max_new_tokens=100,
                                output_scores=True,
                                return_dict_in_generate=True,
                            )
                            gen = self.processor.decode(output[0][0][len(single_input['input_ids'][-1]):], skip_special_tokens=True, output_attentions=True)

                        else:
                            output = self.model.generate(
                                **single_input,
                                keys=keys,
                                weight=weight,
                                adjust_method=adjust_method,
                                layer_weights=None,
                                head_weights=head_weights,
                                head_apply_layers=ablate_head_layers,
                                active_head_mask=active_head_mask,
                                enable_nonsquare_scaling=enable_nonsquare_scaling,
                                text_attn_scale=text_attn_scale,
                                image_attn_scale=image_attn_scale,
                                pos=object_keys,
                                object_attn_scale=object_text_attn_scale,
                                top_flat_fraction=top_flat_fraction,
                                top_flat_mix=top_flat_mix,
                                max_new_tokens=100,
                                output_scores=True,
                                return_dict_in_generate=True,
                            )
                            uncertainty = _compute_uncertainty(
                                output['scores'],
                                "max_prob",
                                token_window=uncertainty_token_window,
                                token_agg=uncertainty_token_agg,
                            )
                            gen = self.processor.decode(output[0][0][len(single_input['input_ids'][-1]):], skip_special_tokens=True, output_attentions=True)
                        
                        # Check correctness of the generated response
                        label = int(batch['labels'][0][idx])
                        if label == 1:
                            TP += 1 if 'Yes' in gen else 0
                            FN += 1 if 'Yes' not in gen else 0
                        else:
                            TN += 1 if 'No' in gen else 0
                            FP += 1 if 'No' not in gen else 0
                        
                        print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
                        
                        # Create result entry for the current sample
                        gold = 'Yes' if label == 1 else 'No'
                        result = {
                            "Prompt": prompt,
                            "Generation": gen,
                            "Golden": gold,
                            "Uncertainty": uncertainty,
                        }
                        results.append(result)
                        index_of_total += 1
                        
                index += 1    
        # Calculate metrics
        precision = TP / (TP + FN)
        recall = TN / (TN + FP)
        f1_score = 2 * precision * recall / (precision + recall)

        print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}\n"
            f"Accuracy: {(TN + TP) / (TN + TP + FN + FP)}\n"
            f"Precision: {precision}\n"
            f"Recall: {recall}\n"
            f"F1 Score: {f1_score}")
        
        all_scores = (TP, TN, FP, FN)
        
        # Save results to JSON file
        output_file_path = f'./outputs/results_{dataset}_{method}_{weight}.json'
        with open(output_file_path, 'w', encoding='utf-8') as fout:
            json.dump(results, fout, ensure_ascii=False, indent=4)
        
        # Save evaluation metrics
        output_score_file = output_file_path.replace(".json", "_scores.json")
        with open(output_score_file, 'w', encoding='utf-8') as fout:
            json.dump({"acc": (TN + TP) / (TN + TP + FN + FP), "precision": precision, "recall": recall, "f1": f1_score}, fout, ensure_ascii=False, indent=4)
        return all_scores
    