"""Deterministic restricted-option scoring for frozen LLaVA-NeXT-Video.

The scorer preserves the public LLaVA-NeXT Qwen conversation template,
``<image>`` placeholder, SigLIP processor, and native ``video`` modality.  It
changes only the decision readout: instead of generating free-form text, the
last language hidden state is projected onto the declared bare option-letter
tokens.  This avoids materializing vocabulary logits for every visual token
while remaining equivalent to the first generation step.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
import time
from typing import Sequence
import warnings


LLAVANEXT_ROOT = Path(os.environ.get("HDEA_LLAVANEXT_ROOT", "LLaVA-NeXT"))
LLAVANEXT_7B = Path(
    os.environ.get("HDEA_LLAVANEXT_MODEL", "LLaVA-NeXT/llava_7B_models")
)
LLAVANEXT_SIGLIP = Path(
    os.environ.get("HDEA_LLAVANEXT_SIGLIP", "LLaVA-NeXT/siglip_model")
)


@dataclass(frozen=True)
class LLaVANeXTMCQScore:
    predicted_index: int
    predicted_letter: str
    candidate_logits: tuple[float, ...]
    candidate_log_probabilities: tuple[float, ...]
    raw_frame_count: int
    temporally_padded_frame_count: int
    unique_frame_count: int
    raw_height: int
    raw_width: int
    raw_input_pixels: int
    post_vision_tokens: int
    language_input_tokens: int
    total_input_tokens: int
    video_grid_thw: tuple[int, int, int]
    decode_seconds: float
    preprocess_seconds: float
    forward_seconds: float
    peak_memory_bytes: int


def build_llavanext_question(
    *,
    timestamps: Sequence[float],
    duration_seconds: float,
    question: str,
    choices: Sequence[str],
) -> str:
    """Build the public video prompt with exact physical timestamps."""
    from .qwen3 import build_mcq_prompt

    stamp_text = ", ".join(f"{float(value):.2f}s" for value in timestamps)
    timing = (
        f"The video lasts for {float(duration_seconds):.2f} seconds. "
        f"The {len(timestamps)} selected frames are in chronological order "
        f"at these timestamps: {stamp_text}."
    )
    return f"<image>\n{timing}\n{build_mcq_prompt(question, choices)}"


def _install_llava_imports_and_local_siglip() -> None:
    """Route the checkpoint's declared SigLIP tower to its local mirror.

    Upstream dispatch checks whether the tower string is a local path before
    checking whether it contains ``siglip`` and consequently misclassifies a
    local SigLIP mirror as CLIP.  The checkpoint itself contains the full
    vision-tower weights.  This narrow routing patch selects the upstream
    SigLIP tower class without altering its computation or parameters.
    """
    root = str(LLAVANEXT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    import llava.model.llava_arch as llava_arch
    from llava.model.multimodal_encoder.builder import (
        build_vision_tower as upstream_build_vision_tower,
    )
    from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower

    local_siglip = str(LLAVANEXT_SIGLIP)

    def build_local_vision_tower(config, **kwargs):
        tower = getattr(
            config, "mm_vision_tower", getattr(config, "vision_tower", None)
        )
        if str(tower) == local_siglip:
            return SigLipVisionTower(
                local_siglip, vision_tower_cfg=config, **kwargs
            )
        return upstream_build_vision_tower(config, **kwargs)

    llava_arch.build_vision_tower = build_local_vision_tower


class LLaVANeXTMCQScorer:
    """One-forward restricted next-token scorer for LLaVA-NeXT-Video-7B."""

    def __init__(
        self,
        device: str = "cuda:0",
        model_path: str | Path = LLAVANEXT_7B,
        siglip_path: str | Path = LLAVANEXT_SIGLIP,
    ):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        if Path(siglip_path).resolve() != LLAVANEXT_SIGLIP.resolve():
            raise ValueError(
                "this audited scorer requires the declared local SigLIP mirror"
            )
        _install_llava_imports_and_local_siglip()

        warnings.filterwarnings(
            "ignore",
            message=r"for vision_model\..*copying from a non-meta parameter.*",
        )

        import torch
        from llava.model.builder import load_pretrained_model

        self.torch = torch
        self.device = torch.device(device)
        self.model_path = str(model_path)
        self.siglip_path = str(siglip_path)
        self.tokenizer, self.model, self.image_processor, self.max_length = (
            load_pretrained_model(
                self.model_path,
                None,
                "llava_qwen",
                torch_dtype="bfloat16",
                device_map={"": str(self.device)},
                overwrite_config={"mm_vision_tower": self.siglip_path},
            )
        )
        self.model.eval()
        meta_parameters = [
            name for name, parameter in self.model.named_parameters()
            if parameter.is_meta
        ]
        if meta_parameters:
            preview = ", ".join(meta_parameters[:5])
            raise RuntimeError(
                f"LLaVA-NeXT has {len(meta_parameters)} unloaded meta parameters: "
                f"{preview}"
            )

        vision = self.model.get_vision_tower()
        if type(vision).__name__ != "SigLipVisionTower":
            raise RuntimeError(
                f"expected SigLipVisionTower, got {type(vision).__name__}"
            )
        self.native_patch_grid = int(vision.num_patches_per_side)
        # ``prepare_inputs_labels_for_multimodal`` calls ``get_2dPool``
        # without a stride argument.  Its public default is 2; some released
        # configs omit the redundant field.
        self.pool_stride = int(
            getattr(self.model.config, "mm_spatial_pool_stride", 2)
        )
        # The checkpoint uses bilinear pooling, whose upstream implementation
        # explicitly applies ceil(grid / stride): 27 -> 14.
        self.pooled_grid = math.ceil(self.native_patch_grid / self.pool_stride)
        self.tokens_per_frame = self.pooled_grid * (self.pooled_grid + 1)
        if self.model.config.mm_newline_position != "grid":
            raise RuntimeError("audited token accounting requires grid newlines")
        if bool(getattr(self.model.config, "add_faster_video", False)):
            raise RuntimeError("checkpoint unexpectedly enables faster-video mixing")

    def _option_token_ids(self, choices: Sequence[str]) -> list[int]:
        letters = [chr(ord("A") + index) for index in range(len(choices))]
        token_ids = []
        for letter in letters:
            ids = self.tokenizer.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"option letter {letter!r} is not one token: {ids}")
            token_ids.append(int(ids[0]))
        return token_ids

    def prepare_inputs(
        self,
        *,
        raw_frames,
        frame_indices: Sequence[int],
        total_num_frames: int,
        fps: float,
        question: str,
        choices: Sequence[str],
    ):
        """Validate one observation and prepare the native LLaVA video input."""
        import numpy as np
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token

        raw = np.asarray(raw_frames)
        requested = [int(value) for value in frame_indices]
        if raw.ndim != 4 or raw.shape[-1] != 3:
            raise ValueError("raw_frames must have shape [frames,height,width,3]")
        if len(raw) != len(requested) or not requested:
            raise ValueError("decoded frames and frame indices must align")
        if requested != sorted(requested):
            raise ValueError("LLaVA-NeXT frames must be serialized chronologically")
        if len(requested) != len(set(requested)):
            raise ValueError("LLaVA-NeXT frame indices must be unique")
        if min(requested) < 0 or max(requested) >= int(total_num_frames):
            raise IndexError("selected frame lies outside decoded video")
        if float(fps) <= 0:
            raise ValueError("fps must be positive")
        if not 2 <= len(choices) <= 26:
            raise ValueError("choice count must lie in [2,26]")

        timestamps = [float(value) / float(fps) for value in requested]
        duration_seconds = float(total_num_frames) / float(fps)
        question_text = build_llavanext_question(
            timestamps=timestamps,
            duration_seconds=duration_seconds,
            question=question,
            choices=choices,
        )
        conversation = deepcopy(conv_templates["qwen_1_5"])
        conversation.messages = []
        conversation.append_message(conversation.roles[0], question_text)
        conversation.append_message(conversation.roles[1], None)
        prompt = conversation.get_prompt()
        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.device)
        image_tensor = self.image_processor.preprocess(
            raw, return_tensors="pt"
        )["pixel_values"].to(device=self.device, dtype=self.torch.bfloat16)
        if image_tensor.ndim != 4 or int(image_tensor.shape[0]) != len(requested):
            raise RuntimeError("LLaVA-NeXT processor changed the temporal budget")
        image_placeholders = int((input_ids == IMAGE_TOKEN_INDEX).sum().item())
        if image_placeholders != 1:
            raise RuntimeError(
                f"expected one image placeholder, found {image_placeholders}"
            )
        return raw, requested, timestamps, prompt, input_ids, image_tensor

    def prepare_multimodal_embeddings(self, input_ids, image_tensor):
        """Use the model's public multimodal insertion primitive."""
        prepared = self.model.prepare_inputs_labels_for_multimodal(
            input_ids,
            None,
            None,
            None,
            None,
            [image_tensor],
            ["video"],
            image_sizes=None,
        )
        _, position_ids, attention_mask, _, inputs_embeds, _ = prepared
        if inputs_embeds is None:
            raise RuntimeError("LLaVA-NeXT did not insert visual embeddings")
        return position_ids, attention_mask, inputs_embeds

    def score_decoded(
        self,
        *,
        raw_frames,
        frame_indices: Sequence[int],
        total_num_frames: int,
        fps: float,
        decode_seconds: float,
        question: str,
        choices: Sequence[str],
    ) -> LLaVANeXTMCQScore:
        torch = self.torch

        preprocess_start = time.perf_counter()
        raw, requested, _, _, input_ids, image_tensor = self.prepare_inputs(
            raw_frames=raw_frames,
            frame_indices=frame_indices,
            total_num_frames=total_num_frames,
            fps=fps,
            question=question,
            choices=choices,
        )
        preprocess_seconds = time.perf_counter() - preprocess_start

        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        forward_start = time.perf_counter()
        with torch.inference_mode():
            position_ids, attention_mask, inputs_embeds = (
                self.prepare_multimodal_embeddings(input_ids, image_tensor)
            )
            outputs = self.model.model(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            option_ids = self._option_token_ids(choices)
            candidate_logits_tensor = self.model.lm_head(
                outputs.last_hidden_state[:, -1, :]
            )[0, option_ids].float()
            candidate_log_probs_tensor = torch.log_softmax(
                candidate_logits_tensor, dim=-1
            )
        torch.cuda.synchronize(self.device)
        forward_seconds = time.perf_counter() - forward_start

        language_tokens = int(input_ids.numel()) - 1
        total_tokens = int(inputs_embeds.shape[1])
        post_vision_tokens = total_tokens - language_tokens
        expected_visual_tokens = len(requested) * self.tokens_per_frame
        if post_vision_tokens != expected_visual_tokens:
            raise RuntimeError(
                "unexpected LLaVA-NeXT visual token count: "
                f"observed={post_vision_tokens} expected={expected_visual_tokens}"
            )
        if total_tokens > int(self.model.config.tokenizer_model_max_length):
            raise RuntimeError("multimodal sequence was truncated")

        logits = tuple(float(value) for value in candidate_logits_tensor.cpu())
        log_probs = tuple(float(value) for value in candidate_log_probs_tensor.cpu())
        predicted_index = min(
            range(len(logits)), key=lambda index: (-logits[index], index)
        )
        return LLaVANeXTMCQScore(
            predicted_index=predicted_index,
            predicted_letter=chr(ord("A") + predicted_index),
            candidate_logits=logits,
            candidate_log_probabilities=log_probs,
            raw_frame_count=len(requested),
            temporally_padded_frame_count=len(requested),
            unique_frame_count=len(requested),
            raw_height=int(raw.shape[1]),
            raw_width=int(raw.shape[2]),
            raw_input_pixels=int(raw.shape[0] * raw.shape[1] * raw.shape[2]),
            post_vision_tokens=post_vision_tokens,
            language_input_tokens=language_tokens,
            total_input_tokens=total_tokens,
            video_grid_thw=(len(requested), self.pooled_grid, self.pooled_grid + 1),
            decode_seconds=float(decode_seconds),
            preprocess_seconds=float(preprocess_seconds),
            forward_seconds=float(forward_seconds),
            peak_memory_bytes=int(torch.cuda.max_memory_allocated(self.device)),
        )
