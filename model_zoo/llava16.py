import warnings

from . import llava15 as _llava15
from .llava import LlavaForConditionalGeneration, LlavaForConditionalGenerationScal
from transformers import AutoProcessor, LlamaTokenizerFast, CLIPImageProcessor

# Candidate checkpoints for LLaVA 1.6.
# We try vicuna first (closest to llama-based stack used by this repo),
# then fallback candidates if loading fails.
_MODEL_CANDIDATES = [
    "llava-hf/llava-v1.6-vicuna-7b-hf",
    "llava-hf/llava-v1.6-mistral-7b-hf",
]


class LlavaWrapper(_llava15.LlavaWrapper):
    """
    LLaVA-1.6 wrapper compatible with the existing AdaptVis codepath.

    This reuses the customized LLaVA-1.5 wrapper implementation while swapping
    the upstream model id to a 1.6 checkpoint. If all 1.6 candidates fail to
    load, an explicit error is raised.
    """

    def __init__(self, root_dir, device, method):
        last_err = None
        for model_name in _MODEL_CANDIDATES:
            try:
                if method in {"scaling_vis", "adapt_vis"}:
                    self.model = LlavaForConditionalGenerationScal.from_pretrained(
                        model_name, cache_dir=root_dir, ignore_mismatched_sizes=True
                    ).eval().to(device)
                else:
                    self.model = LlavaForConditionalGeneration.from_pretrained(
                        model_name, cache_dir=root_dir, ignore_mismatched_sizes=True
                    ).eval().to(device)

                self.feature_extractor = CLIPImageProcessor.from_pretrained(model_name, cache_dir=root_dir)
                self.tokenizer = LlamaTokenizerFast.from_pretrained(model_name, cache_dir=root_dir)
                self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=root_dir)
                self.device = device
                self.model_name = model_name
                return
            except Exception as e:  # noqa: BLE001
                last_err = e
                warnings.warn(f"Failed loading {model_name}: {e}", RuntimeWarning)
        raise RuntimeError(
            "Failed to initialize llava1.6 wrapper with all candidates. "
            f"Last error: {last_err}"
        )
