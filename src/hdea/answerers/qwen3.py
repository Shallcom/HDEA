"""Deterministic one-call multiple-choice scoring with frozen Qwen3-VL."""
from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Sequence


QWEN3_MODEL = os.environ.get("HDEA_QWEN3_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
REMOVED_QUESTION = "[QUESTION REMOVED]"


def top_two_indices(logits: Sequence[float]) -> tuple[int, int]:
    """Return stable top-two candidate indices for a factual decision."""

    if len(logits) < 2:
        raise ValueError("at least two candidate logits are required")
    order = sorted(
        range(len(logits)), key=lambda index: (-float(logits[index]), index)
    )
    return order[0], order[1]


def build_mcq_prompt(
    question: str,
    choices: Sequence[str],
    *,
    timestamped_transcript: str | None = None,
    timestamped_event_memory: str | None = None,
) -> str:
    if timestamped_transcript is not None and timestamped_event_memory is not None:
        raise ValueError("transcript and event memory cannot both be supplied")
    letters = [chr(ord("A") + idx) for idx in range(len(choices))]
    options = "\n".join(f"{letter}. {choice}" for letter, choice in zip(letters, choices))
    allowed = ", ".join(letters[:-1]) + f", or {letters[-1]}"
    transcript = ""
    if timestamped_transcript is not None:
        cleaned = timestamped_transcript.strip()
        if cleaned:
            transcript = (
                "Timestamped transcript excerpts from the video are provided below. "
                "They may be incomplete or irrelevant; use them only when they provide "
                "evidence for the question.\n"
                "<transcript>\n" + cleaned + "\n</transcript>\n"
            )
    event_memory = ""
    if timestamped_event_memory is not None:
        cleaned = timestamped_event_memory.strip()
        if cleaned:
            event_memory = (
                "Question-independent timestamped visual event observations are "
                "provided below. They are generated summaries and may be incomplete "
                "or inaccurate; use them only when they are consistent with the "
                "visible video evidence.\n"
                "<event_memory>\n" + cleaned + "\n</event_memory>\n"
            )
    return (
        transcript
        + event_memory
        + "Question: " + question.strip() + "\n"
        "Options:\n" + options + "\n"
        "Select the best answer using the video. "
        f"Answer with only one option letter: {allowed}."
    )


def build_slowfast_content(
    prompt: str,
    *,
    fast_stream_label: str | None = None,
    slow_stream_label: str | None = None,
    composition_rule: str | None = None,
) -> list[dict[str, str]]:
    """Build either the legacy two-video layout or a frozen role-aware layout."""

    values = (fast_stream_label, slow_stream_label, composition_rule)
    if any(value is not None for value in values) and not all(
        isinstance(value, str) and value.strip() for value in values
    ):
        raise ValueError("all three stream-role strings must be supplied together")
    if fast_stream_label is None:
        return [
            {"type": "video"},
            {"type": "video"},
            {"type": "text", "text": prompt},
        ]
    return [
        {"type": "text", "text": fast_stream_label.strip()},
        {"type": "video"},
        {"type": "text", "text": slow_stream_label.strip()},
        {"type": "video"},
        {"type": "text", "text": composition_rule.strip() + "\n" + prompt},
    ]


def build_text_evidence_mcq_prompt(
    question: str,
    choices: Sequence[str],
    *,
    timestamped_transcript: str,
) -> str:
    """Build the text-only arm without pretending that frames are present."""

    letters = [chr(ord("A") + idx) for idx in range(len(choices))]
    options = "\n".join(f"{letter}. {choice}" for letter, choice in zip(letters, choices))
    allowed = ", ".join(letters[:-1]) + f", or {letters[-1]}"
    return (
        "Timestamped transcript excerpts from the video are provided below. "
        "They may be incomplete or irrelevant; use only the information explicitly "
        "supported by the transcript.\n"
        "<transcript>\n" + timestamped_transcript.strip() + "\n</transcript>\n"
        "Question: " + question.strip() + "\n"
        "Options:\n" + options + "\n"
        "Select the best answer using only the transcript evidence. "
        f"Answer with only one option letter: {allowed}."
    )


@dataclass(frozen=True)
class MCQScore:
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


@dataclass(frozen=True)
class TextMCQScore:
    predicted_index: int
    predicted_letter: str
    candidate_logits: tuple[float, ...]
    candidate_log_probabilities: tuple[float, ...]
    total_input_tokens: int
    forward_seconds: float
    peak_memory_bytes: int


@dataclass(frozen=True)
class SlowFastMCQScore:
    predicted_index: int
    predicted_letter: str
    candidate_logits: tuple[float, ...]
    candidate_log_probabilities: tuple[float, ...]
    fast_raw_frame_count: int
    fast_unique_frame_count: int
    fast_raw_height: int
    fast_raw_width: int
    fast_raw_input_pixels: int
    fast_post_vision_tokens: int
    fast_video_grid_thw: tuple[int, int, int]
    slow_raw_frame_count: int
    slow_unique_frame_count: int
    slow_raw_height: int
    slow_raw_width: int
    slow_raw_input_pixels: int
    slow_post_vision_tokens: int
    slow_video_grid_thw: tuple[int, int, int]
    total_post_vision_tokens: int
    total_input_tokens: int
    decode_seconds: float
    preprocess_seconds: float
    forward_seconds: float
    peak_memory_bytes: int


@dataclass(frozen=True)
class VideoTextGeneration:
    text: str
    raw_frame_count: int
    unique_frame_count: int
    post_vision_tokens: int
    total_input_tokens: int
    generated_tokens: int
    video_grid_thw: tuple[int, int, int]
    preprocess_seconds: float
    generation_seconds: float
    peak_memory_bytes: int


class Qwen3MCQScorer:
    def __init__(
        self,
        device: str = "cuda:0",
        model_path: str = QWEN3_MODEL,
        adapter_path: str | None = None,
    ):
        import json
        from pathlib import Path

        import torch
        from transformers import AutoProcessor

        self.torch = torch
        self.device = torch.device(device)
        config_path = Path(model_path) / "config.json"
        with config_path.open(encoding="utf-8") as handle:
            model_type = str(json.load(handle).get("model_type", ""))
        if model_type == "qwen3_vl":
            from transformers import Qwen3VLForConditionalGeneration

            model_class = Qwen3VLForConditionalGeneration
            processor_kwargs = {}
        elif model_type == "qwen2_5_vl":
            from transformers import Qwen2_5_VLForConditionalGeneration

            model_class = Qwen2_5_VLForConditionalGeneration
            # Match the public DIG/LMMs-Eval visual budget.  Newer fast
            # processors otherwise upscale low-resolution MLVU frames and can
            # silently create >40k-token examples from the same 32 frames.
            processor_kwargs = {"min_pixels": 25088, "max_pixels": 100352}
        else:
            raise ValueError(
                f"unsupported Qwen-VL model type {model_type!r} at {model_path}"
            )
        self.model_type = model_type
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            **processor_kwargs,
        )
        model = model_class.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        if adapter_path is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model,
                adapter_path,
                local_files_only=True,
                is_trainable=False,
            )
        self.model = model.to(self.device).eval()

    def _option_token_ids(self, choices: Sequence[str]) -> list[int]:
        letters = [chr(ord("A") + idx) for idx in range(len(choices))]
        option_token_ids = []
        for letter in letters:
            token_ids = self.processor.tokenizer.encode(letter, add_special_tokens=False)
            if len(token_ids) != 1:
                raise RuntimeError(f"option letter {letter!r} is not one token: {token_ids}")
            option_token_ids.append(token_ids[0])
        return option_token_ids

    def _expand_video_placeholders(self, text: str, grids, metadata_rows) -> str:
        """Mirror Qwen3VLProcessor's official timestamped-video expansion."""

        merge_length = self.processor.video_processor.merge_size**2
        expanded = str(text)
        for grid, metadata in zip(grids, metadata_rows):
            if self.processor.video_token not in expanded:
                raise RuntimeError("video placeholder count is smaller than video streams")
            timestamps = self.processor._calculate_timestamps(
                metadata.frames_indices,
                metadata.fps,
                self.processor.video_processor.merge_size,
            )
            frame_count = int(grid[0])
            if len(timestamps) != frame_count:
                raise RuntimeError("timestamp and temporal-grid counts differ")
            frame_seqlen = int(grid[1:].prod().item()) // merge_length
            placeholder = ""
            for timestamp in timestamps:
                placeholder += f"<{timestamp:.1f} seconds>"
                placeholder += self.processor.vision_start_token
                placeholder += "<|placeholder|>" * frame_seqlen
                placeholder += self.processor.vision_end_token
            wrapped = (
                self.processor.vision_start_token
                + self.processor.video_token
                + self.processor.vision_end_token
            )
            if wrapped in expanded:
                expanded = expanded.replace(wrapped, placeholder, 1)
            else:
                expanded = expanded.replace(self.processor.video_token, placeholder, 1)
        if self.processor.video_token in expanded:
            raise RuntimeError("video placeholder count is larger than video streams")
        return expanded.replace("<|placeholder|>", self.processor.video_token)

    def score_slowfast_decoded(
        self,
        *,
        fast_raw_frames,
        fast_frame_indices: Sequence[int],
        slow_raw_frames,
        slow_frame_indices: Sequence[int],
        total_num_frames: int,
        fps: float,
        decode_seconds: float,
        question: str,
        choices: Sequence[str],
        fast_pixels_per_raw_frame_ceiling: int = 16384,
        timestamped_event_memory: str | None = None,
        allow_fast_duplicates: bool = False,
        fast_stream_label: str | None = None,
        slow_stream_label: str | None = None,
        composition_rule: str | None = None,
    ) -> SlowFastMCQScore:
        """Score a compressed unseen-frame stream followed by protected Slow.

        The Slow stream uses the exact native preprocessing of ``score_decoded``.
        Fast uses the same frozen vision encoder but a smaller, aspect-preserving
        pixel ceiling.  Both streams enter one language-model forward call.
        """

        import numpy as np
        import torch

        fast_raw = np.asarray(fast_raw_frames)
        slow_raw = np.asarray(slow_raw_frames)
        fast_requested = [int(value) for value in fast_frame_indices]
        slow_requested = [int(value) for value in slow_frame_indices]
        for name, raw, requested in (
            ("fast", fast_raw, fast_requested),
            ("slow", slow_raw, slow_requested),
        ):
            if raw.ndim != 4 or raw.shape[-1] != 3:
                raise ValueError(f"{name}_raw_frames must have shape [frames,height,width,3]")
            if len(raw) != len(requested) or not requested:
                raise ValueError(f"{name} decoded frames and indices must align")
            if min(requested) < 0 or max(requested) >= int(total_num_frames):
                raise IndexError(f"{name} selected frame lies outside the video")
            if requested != sorted(requested):
                raise ValueError(f"{name} frame indices must be chronological")
            if name == "slow" and len(requested) != len(set(requested)):
                raise ValueError("slow frame indices must be unique")
            if name == "fast" and not allow_fast_duplicates and len(requested) != len(set(requested)):
                raise ValueError("fast frame indices must be unique unless explicitly enabled")
        if set(fast_requested) & set(slow_requested):
            raise ValueError("Fast frames must be disjoint from protected Slow frames")
        if fast_pixels_per_raw_frame_ceiling <= 0:
            raise ValueError("Fast pixel ceiling must be positive")
        if not 2 <= len(choices) <= 26:
            raise ValueError("choice count must be between 2 and 26")

        def metadata(raw, requested):
            return {
                "total_num_frames": int(total_num_frames),
                "fps": float(fps),
                "duration": int(total_num_frames) / float(fps),
                "frames_indices": requested.copy(),
                "height": int(raw.shape[1]),
                "width": int(raw.shape[2]),
            }

        fast_video = torch.from_numpy(fast_raw).permute(0, 3, 1, 2).contiguous()
        slow_video = torch.from_numpy(slow_raw).permute(0, 3, 1, 2).contiguous()
        preprocess_start = time.perf_counter()
        fast_inputs = self.processor.video_processor(
            videos=[fast_video],
            video_metadata=[metadata(fast_raw, fast_requested)],
            do_sample_frames=False,
            size={
                "shortest_edge": 4096,
                "longest_edge": int(len(fast_requested) * fast_pixels_per_raw_frame_ceiling),
            },
            return_metadata=True,
            return_tensors="pt",
        )
        slow_inputs = self.processor.video_processor(
            videos=[slow_video],
            video_metadata=[metadata(slow_raw, slow_requested)],
            do_sample_frames=False,
            return_metadata=True,
            return_tensors="pt",
        )
        video_grid_thw = torch.cat(
            [fast_inputs["video_grid_thw"], slow_inputs["video_grid_thw"]], dim=0
        )
        pixel_values_videos = torch.cat(
            [
                fast_inputs["pixel_values_videos"],
                slow_inputs["pixel_values_videos"],
            ],
            dim=0,
        )
        prompt = build_mcq_prompt(
            question,
            choices,
            timestamped_event_memory=timestamped_event_memory,
        )
        messages = [{
            "role": "user",
            "content": build_slowfast_content(
                prompt,
                fast_stream_label=fast_stream_label,
                slow_stream_label=slow_stream_label,
                composition_rule=composition_rule,
            ),
        }]
        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        expanded_text = self._expand_video_placeholders(
            chat_text,
            video_grid_thw,
            fast_inputs["video_metadata"] + slow_inputs["video_metadata"],
        )
        text_inputs = self.processor.tokenizer([expanded_text], return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in text_inputs.items()}
        inputs["pixel_values_videos"] = pixel_values_videos.to(
            self.device, dtype=torch.bfloat16
        )
        inputs["video_grid_thw"] = video_grid_thw.to(self.device)
        preprocess_seconds = time.perf_counter() - preprocess_start

        fast_grid = tuple(int(value) for value in video_grid_thw[0].tolist())
        slow_grid = tuple(int(value) for value in video_grid_thw[1].tolist())
        merge_length = self.processor.video_processor.merge_size**2
        fast_tokens = int(np.prod(fast_grid) // merge_length)
        slow_tokens = int(np.prod(slow_grid) // merge_length)
        video_token_count = int(
            (inputs["input_ids"] == self.processor.video_token_id).sum().item()
        )
        if video_token_count != fast_tokens + slow_tokens:
            raise RuntimeError("expanded prompt and visual-grid token counts differ")

        option_token_ids = self._option_token_ids(choices)
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        forward_start = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(**inputs, use_cache=False, logits_to_keep=1)
            candidate_logits = outputs.logits[0, -1, option_token_ids].float()
            candidate_log_probabilities = torch.log_softmax(candidate_logits, dim=0)
        torch.cuda.synchronize(self.device)
        forward_seconds = time.perf_counter() - forward_start
        predicted_index = int(candidate_logits.argmax().item())
        letters = [chr(ord("A") + idx) for idx in range(len(choices))]
        return SlowFastMCQScore(
            predicted_index=predicted_index,
            predicted_letter=letters[predicted_index],
            candidate_logits=tuple(float(value) for value in candidate_logits.cpu().tolist()),
            candidate_log_probabilities=tuple(
                float(value) for value in candidate_log_probabilities.cpu().tolist()
            ),
            fast_raw_frame_count=len(fast_raw),
            fast_unique_frame_count=len(set(fast_requested)),
            fast_raw_height=int(fast_raw.shape[1]),
            fast_raw_width=int(fast_raw.shape[2]),
            fast_raw_input_pixels=int(np.prod(fast_raw.shape[:3])),
            fast_post_vision_tokens=fast_tokens,
            fast_video_grid_thw=fast_grid,
            slow_raw_frame_count=len(slow_raw),
            slow_unique_frame_count=len(set(slow_requested)),
            slow_raw_height=int(slow_raw.shape[1]),
            slow_raw_width=int(slow_raw.shape[2]),
            slow_raw_input_pixels=int(np.prod(slow_raw.shape[:3])),
            slow_post_vision_tokens=slow_tokens,
            slow_video_grid_thw=slow_grid,
            total_post_vision_tokens=fast_tokens + slow_tokens,
            total_input_tokens=int(inputs["input_ids"].shape[1]),
            decode_seconds=float(decode_seconds),
            preprocess_seconds=preprocess_seconds,
            forward_seconds=forward_seconds,
            peak_memory_bytes=int(torch.cuda.max_memory_allocated(self.device)),
        )

    def score_text(
        self,
        *,
        question: str,
        choices: Sequence[str],
        timestamped_transcript: str | None = None,
    ) -> TextMCQScore:
        """Score the identical MCQ prompt without a visual observation."""
        import torch

        if not 2 <= len(choices) <= 26:
            raise ValueError("choice count must be between 2 and 26")
        prompt = (
            build_mcq_prompt(question, choices)
            if timestamped_transcript is None
            else build_text_evidence_mcq_prompt(
                question,
                choices,
                timestamped_transcript=timestamped_transcript,
            )
        )
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
        option_token_ids = self._option_token_ids(choices)

        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        forward_start = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(**inputs, use_cache=False, logits_to_keep=1)
            candidate_logits = outputs.logits[0, -1, option_token_ids].float()
            candidate_log_probabilities = torch.log_softmax(candidate_logits, dim=0)
        torch.cuda.synchronize(self.device)
        forward_seconds = time.perf_counter() - forward_start
        predicted_index = int(candidate_logits.argmax().item())
        letters = [chr(ord("A") + idx) for idx in range(len(choices))]
        return TextMCQScore(
            predicted_index=predicted_index,
            predicted_letter=letters[predicted_index],
            candidate_logits=tuple(float(value) for value in candidate_logits.cpu().tolist()),
            candidate_log_probabilities=tuple(
                float(value) for value in candidate_log_probabilities.cpu().tolist()
            ),
            total_input_tokens=int(inputs["input_ids"].shape[1]),
            forward_seconds=forward_seconds,
            peak_memory_bytes=int(torch.cuda.max_memory_allocated(self.device)),
        )

    def generate_video_text_decoded(
        self,
        *,
        raw_frames,
        frame_indices: Sequence[int],
        total_num_frames: int,
        fps: float,
        prompt: str,
        max_new_tokens: int = 192,
        pixels_per_raw_frame_ceiling: int = 65536,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int | None = None,
        enable_thinking: bool | None = None,
    ) -> VideoTextGeneration:
        """Generate bounded text from timestamped decoded video observations."""

        import numpy as np
        import torch

        raw = np.asarray(raw_frames)
        requested = [int(value) for value in frame_indices]
        if raw.ndim != 4 or raw.shape[-1] != 3:
            raise ValueError("raw_frames must have shape [frames,height,width,3]")
        if len(raw) != len(requested) or not requested:
            raise ValueError("decoded frames and indices must align")
        if requested != sorted(requested) or len(requested) != len(set(requested)):
            raise ValueError("frame indices must be unique and chronological")
        if min(requested) < 0 or max(requested) >= int(total_num_frames):
            raise IndexError("selected frame lies outside the video")
        if max_new_tokens <= 0 or pixels_per_raw_frame_ceiling <= 0:
            raise ValueError("generation budgets must be positive")
        if temperature <= 0.0 or not 0.0 < top_p <= 1.0:
            raise ValueError("sampling parameters are outside their valid range")

        # Qwen3-VL's temporal patch size is two and its low-level video
        # processor rejects t=1 before applying ordinary odd-length padding.
        # Duplicate a singleton only at serialization time; provenance and
        # unique-observation counts continue to describe the one real frame.
        serialized_raw = raw
        serialized_requested = requested.copy()
        if len(requested) == 1:
            serialized_raw = np.concatenate([raw, raw], axis=0)
            serialized_requested = [requested[0], requested[0]]
        video = torch.from_numpy(serialized_raw).permute(0, 3, 1, 2).contiguous()
        metadata = {
            "total_num_frames": int(total_num_frames),
            "fps": float(fps),
            "duration": int(total_num_frames) / float(fps),
            "frames_indices": serialized_requested.copy(),
            "height": int(serialized_raw.shape[1]),
            "width": int(serialized_raw.shape[2]),
        }
        preprocess_start = time.perf_counter()
        video_inputs = self.processor.video_processor(
            videos=[video],
            video_metadata=[metadata],
            do_sample_frames=False,
            size={
                "shortest_edge": 4096,
                "longest_edge": int(len(serialized_requested) * pixels_per_raw_frame_ceiling),
            },
            return_metadata=True,
            return_tensors="pt",
        )
        grid = video_inputs["video_grid_thw"]
        messages = [{
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": str(prompt)},
            ],
        }]
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = bool(enable_thinking)
        chat_text = self.processor.apply_chat_template(messages, **template_kwargs)
        expanded_text = self._expand_video_placeholders(
            chat_text, grid, video_inputs["video_metadata"]
        )
        text_inputs = self.processor.tokenizer([expanded_text], return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in text_inputs.items()}
        inputs["pixel_values_videos"] = video_inputs["pixel_values_videos"].to(
            self.device, dtype=torch.bfloat16
        )
        inputs["video_grid_thw"] = grid.to(self.device)
        preprocess_seconds = time.perf_counter() - preprocess_start
        merge_length = self.processor.video_processor.merge_size**2
        grid_tuple = tuple(int(value) for value in grid[0].tolist())
        vision_tokens = int(np.prod(grid_tuple) // merge_length)
        observed_tokens = int(
            (inputs["input_ids"] == self.processor.video_token_id).sum().item()
        )
        if observed_tokens != vision_tokens:
            raise RuntimeError("expanded prompt and visual-grid token counts differ")

        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        generation_start = time.perf_counter()
        with torch.inference_mode():
            if seed is not None:
                torch.manual_seed(int(seed))
                torch.cuda.manual_seed_all(int(seed))
            generated = self.model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=bool(do_sample),
                temperature=float(temperature) if do_sample else None,
                top_p=float(top_p) if do_sample else None,
                use_cache=True,
            )
        torch.cuda.synchronize(self.device)
        generation_seconds = time.perf_counter() - generation_start
        input_length = int(inputs["input_ids"].shape[1])
        new_ids = generated[0, input_length:]
        text = self.processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return VideoTextGeneration(
            text=text,
            raw_frame_count=len(serialized_requested),
            unique_frame_count=len(set(requested)),
            post_vision_tokens=vision_tokens,
            total_input_tokens=input_length,
            generated_tokens=int(new_ids.numel()),
            video_grid_thw=grid_tuple,
            preprocess_seconds=preprocess_seconds,
            generation_seconds=generation_seconds,
            peak_memory_bytes=int(torch.cuda.max_memory_allocated(self.device)),
        )

    def score(
        self,
        *,
        video_path: str,
        frame_indices: Sequence[int],
        question: str,
        choices: Sequence[str],
        timestamped_transcript: str | None = None,
    ) -> MCQScore:
        from decord import VideoReader, cpu

        if not frame_indices:
            raise ValueError("at least one video frame is required")
        if not 2 <= len(choices) <= 26:
            raise ValueError("choice count must be between 2 and 26")

        reader = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        return self.score_reader(
            reader=reader,
            frame_indices=frame_indices,
            question=question,
            choices=choices,
            timestamped_transcript=timestamped_transcript,
        )

    def score_reader(
        self,
        *,
        reader,
        frame_indices: Sequence[int],
        question: str,
        choices: Sequence[str],
        timestamped_transcript: str | None = None,
    ) -> MCQScore:
        """Score frames from an already open Decord reader."""

        if not frame_indices:
            raise ValueError("at least one video frame is required")
        if not 2 <= len(choices) <= 26:
            raise ValueError("choice count must be between 2 and 26")
        decode_start = time.perf_counter()
        requested = [int(value) for value in frame_indices]
        if min(requested) < 0 or max(requested) >= len(reader):
            raise IndexError("selected frame index lies outside the decoded video")
        raw = reader.get_batch(requested).asnumpy()
        fps = float(reader.get_avg_fps())
        decode_seconds = time.perf_counter() - decode_start

        return self.score_decoded(
            raw_frames=raw,
            frame_indices=requested,
            total_num_frames=len(reader),
            fps=fps,
            decode_seconds=decode_seconds,
            question=question,
            choices=choices,
            timestamped_transcript=timestamped_transcript,
        )

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
        timestamped_transcript: str | None = None,
    ) -> MCQScore:
        """Score already decoded RGB frames with the identical model path.

        This entry point allows several evidence arms for the same question to
        share one physical video decode.  It changes neither selected pixels,
        their temporal order, processor metadata, prompt, nor model execution.
        """
        import numpy as np
        import torch

        raw = np.asarray(raw_frames)
        requested = [int(value) for value in frame_indices]
        if raw.ndim != 4 or raw.shape[-1] != 3:
            raise ValueError("raw_frames must have shape [frames, height, width, 3]")
        if len(raw) != len(requested) or not requested:
            raise ValueError("decoded frames and frame indices must be aligned")
        if min(requested) < 0 or max(requested) >= int(total_num_frames):
            raise IndexError("selected frame index lies outside the decoded video")
        if not 2 <= len(choices) <= 26:
            raise ValueError("choice count must be between 2 and 26")
        decoded_raw_frame_count = int(raw.shape[0])
        video = torch.from_numpy(raw).permute(0, 3, 1, 2).contiguous()
        if self.model_type == "qwen2_5_vl" and int(raw.shape[1] * raw.shape[2]) > 100352:
            # ``lmms-eval``/qwen-vl-utils applies this ceiling while loading
            # each frame.  Our runner starts from already decoded tensors, so
            # perform the equivalent aspect-preserving, patch-aligned resize
            # explicitly before handing frames to the HF processor.
            import math
            import torch.nn.functional as functional

            factor = 28
            scale = math.sqrt(100352.0 / float(raw.shape[1] * raw.shape[2]))
            target_h = max(factor, int(round(raw.shape[1] * scale / factor)) * factor)
            target_w = max(factor, int(round(raw.shape[2] * scale / factor)) * factor)
            while target_h * target_w > 100352:
                if target_h >= target_w and target_h > factor:
                    target_h -= factor
                elif target_w > factor:
                    target_w -= factor
                else:
                    break
            video = functional.interpolate(
                video.float(),
                size=(target_h, target_w),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).round().clamp_(0, 255).to(torch.uint8)

        prompt = build_mcq_prompt(
            question,
            choices,
            timestamped_transcript=timestamped_transcript,
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        metadata = {
            "total_num_frames": int(total_num_frames),
            "fps": fps,
            "duration": int(total_num_frames) / fps,
            # The processor pads odd temporal sequences and mutates the
            # metadata list in-place while constructing patch timestamps.
            # Keep the acquisition list itself immutable for audit logging.
            "frames_indices": requested.copy(),
            "height": int(raw.shape[1]),
            "width": int(raw.shape[2]),
        }

        preprocess_start = time.perf_counter()
        processor_kwargs = {}
        if self.model_type == "qwen2_5_vl":
            # Qwen2.5's processor reads the aggregate video pixel ceiling
            # from ``video_processor.size``.  Passing max_pixels only at
            # construction time updates a separate per-image field in recent
            # Transformers releases and leaves this 12.8M default untouched.
            self.processor.video_processor.size = {
                "shortest_edge": 3136,
                "longest_edge": int(len(requested) * 100352),
            }
        inputs = self.processor(
            text=[text],
            videos=[video],
            video_metadata=[metadata],
            do_sample_frames=False,
            return_tensors="pt",
            **processor_kwargs,
        )
        post_vision_tokens = int(
            (inputs["input_ids"] == self.processor.video_token_id).sum().item()
        )
        total_input_tokens = int(inputs["input_ids"].shape[1])
        video_grid = tuple(int(value) for value in inputs["video_grid_thw"][0].tolist())
        temporally_padded_frame_count = int(video_grid[0] * 2)
        inputs = inputs.to(self.device)
        inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(torch.bfloat16)
        preprocess_seconds = time.perf_counter() - preprocess_start

        letters = [chr(ord("A") + idx) for idx in range(len(choices))]
        option_token_ids = self._option_token_ids(choices)

        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        forward_start = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(**inputs, use_cache=False, logits_to_keep=1)
            candidate_logits = outputs.logits[0, -1, option_token_ids].float()
            candidate_log_probabilities = torch.log_softmax(candidate_logits, dim=0)
        torch.cuda.synchronize(self.device)
        forward_seconds = time.perf_counter() - forward_start
        peak_memory = int(torch.cuda.max_memory_allocated(self.device))
        predicted_index = int(candidate_logits.argmax().item())

        return MCQScore(
            predicted_index=predicted_index,
            predicted_letter=letters[predicted_index],
            candidate_logits=tuple(float(value) for value in candidate_logits.cpu().tolist()),
            candidate_log_probabilities=tuple(
                float(value) for value in candidate_log_probabilities.cpu().tolist()
            ),
            raw_frame_count=decoded_raw_frame_count,
            temporally_padded_frame_count=temporally_padded_frame_count,
            unique_frame_count=len(set(requested)),
            raw_height=int(video.shape[2]),
            raw_width=int(video.shape[3]),
            raw_input_pixels=int(video.shape[0] * video.shape[2] * video.shape[3]),
            post_vision_tokens=post_vision_tokens,
            total_input_tokens=total_input_tokens,
            video_grid_thw=video_grid,
            decode_seconds=decode_seconds,
            preprocess_seconds=preprocess_seconds,
            forward_seconds=forward_seconds,
            peak_memory_bytes=peak_memory,
        )
