"""Deterministic restricted-option scoring for frozen mPLUG-Owl3 video MCQ.

The implementation keeps mPLUG-Owl3's native ``<|video|>`` processor and
HyperAttention visual-KV path.  It differs from the public generation example
only at the final decision: the first-step logits are restricted to the
declared bare option letters instead of generating and heuristically parsing
free-form text.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import os
import time
from typing import Sequence


MPLUGOWL3_7B = os.environ.get(
    "HDEA_MPLUGOWL3_MODEL", "mPLUG/mPLUG-Owl3-7B-241101"
)


@dataclass(frozen=True)
class MPLUGOwl3MCQScore:
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


def build_mplugowl3_user_content(
    timestamps: Sequence[float], question: str, choices: Sequence[str]
) -> str:
    """Build the native video message while preserving physical frame time."""
    from .qwen3 import build_mcq_prompt

    stamp_text = ", ".join(f"{float(value):.1f}" for value in timestamps)
    timing = (
        f"The {len(timestamps)} video frames are in chronological order at "
        f"these timestamps in seconds: {stamp_text}."
    )
    return f"<|video|>\n{timing}\n{build_mcq_prompt(question, choices)}"


class MPLUGOwl3MCQScorer:
    """One-forward restricted next-token scorer for mPLUG-Owl3-7B."""

    def __init__(self, device: str = "cuda:0", model_path: str = MPLUGOWL3_7B):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        import torch
        from modelscope import AutoConfig, AutoModel, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device)
        self.model_path = str(model_path)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.processor = self.model.init_processor(self.tokenizer)
        self.vision_patch_size = int(config.vision_config.patch_size)
        self.vision_image_size = int(config.vision_config.image_size)

    def _option_token_ids(self, choices: Sequence[str]) -> list[int]:
        letters = [chr(ord("A") + index) for index in range(len(choices))]
        result = []
        for letter in letters:
            ids = self.tokenizer.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"option letter {letter!r} is not one token: {ids}")
            result.append(int(ids[0]))
        return result

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
        """Validate and preprocess one selected observation with the native API."""
        import numpy as np
        from PIL import Image

        raw = np.asarray(raw_frames)
        requested = [int(value) for value in frame_indices]
        if raw.ndim != 4 or raw.shape[-1] != 3:
            raise ValueError("raw_frames must have shape [frames,height,width,3]")
        if len(raw) != len(requested) or not requested:
            raise ValueError("decoded frames and frame indices must align")
        if requested != sorted(requested):
            raise ValueError("mPLUG-Owl3 frames must be serialized chronologically")
        if len(requested) != len(set(requested)):
            raise ValueError("mPLUG-Owl3 frame indices must be unique")
        if min(requested) < 0 or max(requested) >= int(total_num_frames):
            raise IndexError("selected frame lies outside decoded video")
        if float(fps) <= 0:
            raise ValueError("fps must be positive")
        if not 2 <= len(choices) <= 26:
            raise ValueError("choice count must lie in [2,26]")

        timestamps = [float(value) / float(fps) for value in requested]
        frames = [Image.fromarray(frame.astype("uint8")).convert("RGB") for frame in raw]
        messages = [
            {
                "role": "user",
                "content": build_mplugowl3_user_content(
                    timestamps, question, choices
                ),
            },
            {"role": "assistant", "content": ""},
        ]
        # The bundled processor expands <|video|> in place.  Keep caller-owned
        # messages immutable so repeated scale scoring cannot inherit expanded
        # media tokens from an earlier observation.
        inputs = self.processor(
            deepcopy(messages), images=None, videos=[frames], cut_enable=False
        )
        inputs.to(self.device)
        media_offsets = inputs["media_offset"]
        if len(media_offsets) != 1 or int(media_offsets[0].numel()) != len(requested):
            raise RuntimeError(
                "mPLUG-Owl3 media offsets do not match the selected frame count"
            )
        if int(inputs["pixel_values"].shape[0]) != len(requested):
            raise RuntimeError(
                "mPLUG-Owl3 processor changed the declared temporal frame budget"
            )
        return raw, requested, timestamps, inputs

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
    ) -> MPLUGOwl3MCQScore:
        import torch

        preprocess_start = time.perf_counter()
        raw, requested, _, inputs = self.prepare_inputs(
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
            image_embeds = self.model.forward_image(inputs["pixel_values"])
            outputs = self.model.language_model(
                input_ids=inputs["input_ids"],
                image_embeds=image_embeds,
                media_offset=inputs["media_offset"],
                use_cache=False,
                return_dict=True,
            )
            option_ids = self._option_token_ids(choices)
            candidate_logits_tensor = outputs.logits[0, -1, option_ids].float()
            candidate_log_probs_tensor = torch.log_softmax(
                candidate_logits_tensor, dim=-1
            )
        torch.cuda.synchronize(self.device)
        forward_seconds = time.perf_counter() - forward_start

        logits = tuple(float(value) for value in candidate_logits_tensor.cpu().tolist())
        log_probs = tuple(
            float(value) for value in candidate_log_probs_tensor.cpu().tolist()
        )
        predicted_index = min(
            range(len(logits)), key=lambda index: (-logits[index], index)
        )
        tokens_per_frame = int(image_embeds.shape[1])
        post_vision_tokens = int(image_embeds.shape[0]) * tokens_per_frame
        language_tokens = int(inputs["input_ids"].shape[1])
        grid_side = self.vision_image_size // self.vision_patch_size
        if tokens_per_frame != grid_side * grid_side:
            raise RuntimeError(
                f"unexpected mPLUG-Owl3 visual token count: {tokens_per_frame}"
            )
        return MPLUGOwl3MCQScore(
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
            total_input_tokens=language_tokens + post_vision_tokens,
            video_grid_thw=(len(requested), grid_side, grid_side),
            decode_seconds=float(decode_seconds),
            preprocess_seconds=float(preprocess_seconds),
            forward_seconds=float(forward_seconds),
            peak_memory_bytes=int(torch.cuda.max_memory_allocated(self.device)),
        )
