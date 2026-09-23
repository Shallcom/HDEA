"""Deterministic restricted-option scoring for frozen InternVL3 video MCQ.

The implementation mirrors InternVL's official multi-image/video interface,
but reads the next-token logits directly instead of parsing generated text.
Each selected video frame is represented by exactly one 448 x 448 tile and is
paired with its physical timestamp in the textual frame prefix.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Sequence


INTERNVL3_8B = os.environ.get(
    "HDEA_INTERNVL3_MODEL", "OpenGVLab/InternVL3-8B-Instruct"
)


@dataclass(frozen=True)
class InternVL3MCQScore:
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
    total_input_tokens: int
    video_grid_thw: tuple[int, int, int]
    decode_seconds: float
    preprocess_seconds: float
    forward_seconds: float
    peak_memory_bytes: int


class InternVL3MCQScorer:
    """One-forward restricted next-token scorer for InternVL3."""

    def __init__(self, device: str = "cuda:0", model_path: str = INTERNVL3_8B):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device)
        self.model_path = str(model_path)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            local_files_only=True,
        ).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True,
        )
        # The bundled tokenizer advertises 12,288 tokens, while the actual
        # Qwen2.5 language stack in this checkpoint declares and implements a
        # 32,768-position context.  Use the model contract so 48 one-tile
        # frames plus the MCQ prompt do not trigger a stale tokenizer warning.
        self.tokenizer.model_max_length = int(
            self.model.language_model.config.max_position_embeddings
        )
        self.img_context_token = "<IMG_CONTEXT>"
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(
            self.img_context_token
        )
        self.model.img_context_token_id = self.img_context_token_id

    def _option_token_ids(self, choices: Sequence[str]) -> list[int]:
        letters = [chr(ord("A") + index) for index in range(len(choices))]
        result = []
        for letter in letters:
            ids = self.tokenizer.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"option letter {letter!r} is not one token: {ids}")
            result.append(int(ids[0]))
        return result

    def _build_query(
        self,
        *,
        timestamps: Sequence[float],
        question: str,
        choices: Sequence[str],
    ) -> str:
        from .qwen3 import build_mcq_prompt

        frame_prefix = "".join(
            f"Frame {index + 1} at {float(timestamp):.1f} seconds: <image>\n"
            for index, timestamp in enumerate(timestamps)
        )
        prompt = frame_prefix + build_mcq_prompt(question, choices)
        template = self.model.conv_template.copy()
        template.system_message = self.model.system_message
        template.append_message(template.roles[0], prompt)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        image_tokens = (
            "<img>"
            + self.img_context_token * int(self.model.num_image_token)
            + "</img>"
        )
        for _ in timestamps:
            if "<image>" not in query:
                raise RuntimeError("InternVL prompt has fewer image placeholders than frames")
            query = query.replace("<image>", image_tokens, 1)
        if "<image>" in query:
            raise RuntimeError("InternVL prompt has more image placeholders than frames")
        return query

    @staticmethod
    def _transform_frames(raw_frames):
        import numpy as np
        from PIL import Image
        import torch
        from torchvision.transforms import Compose, InterpolationMode, Lambda, Normalize, Resize, ToTensor

        raw = np.asarray(raw_frames)
        transform = Compose([
            Lambda(lambda image: image.convert("RGB")),
            Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            ToTensor(),
            Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])
        return torch.stack([
            transform(Image.fromarray(frame).convert("RGB")) for frame in raw
        ])

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
    ) -> InternVL3MCQScore:
        import numpy as np
        import torch

        raw = np.asarray(raw_frames)
        requested = [int(value) for value in frame_indices]
        if raw.ndim != 4 or raw.shape[-1] != 3:
            raise ValueError("raw_frames must have shape [frames,height,width,3]")
        if len(raw) != len(requested) or not requested:
            raise ValueError("decoded frames and frame indices must align")
        if requested != sorted(requested):
            raise ValueError("InternVL frames must be serialized chronologically")
        if len(requested) != len(set(requested)):
            raise ValueError("InternVL frame indices must be unique")
        if min(requested) < 0 or max(requested) >= int(total_num_frames):
            raise IndexError("selected frame lies outside decoded video")
        if not 2 <= len(choices) <= 26:
            raise ValueError("choice count must lie in [2,26]")
        timestamps = [float(value) / float(fps) for value in requested]

        preprocess_start = time.perf_counter()
        pixel_values = self._transform_frames(raw).to(
            self.device, dtype=torch.bfloat16
        )
        query = self._build_query(
            timestamps=timestamps,
            question=question,
            choices=choices,
        )
        inputs = self.tokenizer(query, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        observed_context = int((input_ids == self.img_context_token_id).sum().item())
        expected_context = len(requested) * int(self.model.num_image_token)
        if observed_context != expected_context:
            raise RuntimeError(
                f"visual placeholder mismatch observed={observed_context} "
                f"expected={expected_context}"
            )
        preprocess_seconds = time.perf_counter() - preprocess_start

        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        forward_start = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_flags=torch.ones(
                    (len(requested), 1), dtype=torch.long, device=self.device
                ),
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
        return InternVL3MCQScore(
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
            post_vision_tokens=expected_context,
            total_input_tokens=int(input_ids.shape[1]),
            video_grid_thw=(len(requested), 1, 1),
            decode_seconds=float(decode_seconds),
            preprocess_seconds=float(preprocess_seconds),
            forward_seconds=float(forward_seconds),
            peak_memory_bytes=int(torch.cuda.max_memory_allocated(self.device)),
        )
