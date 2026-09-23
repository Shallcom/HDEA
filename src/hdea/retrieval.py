"""Frozen SigLIP retrieval wrapper used to build the one-FPS catalog."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np


os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

DEFAULT_RETRIEVER = "google/siglip-so400m-patch14-384"


def generic_retrieval_text(question: str) -> str:
    return f"Question: {question}"


def candidate_retrieval_text(question: str, choice: str) -> str:
    # The hypothesis is placed first so truncation cannot erase the option.
    return f"Candidate answer: {choice}. Question: {question}"


class FrozenSiglip:
    """Expose normalized frozen SigLIP image and text embeddings."""

    def __init__(
        self,
        device: str = "cuda:0",
        model_path: str | Path = DEFAULT_RETRIEVER,
        *,
        local_files_only: bool = False,
    ) -> None:
        import torch
        from transformers import (
            AutoModel,
            SiglipImageProcessor,
            SiglipProcessor,
            SiglipTokenizer,
        )

        self.torch = torch
        self.device = torch.device(device)
        path = str(model_path)
        image_processor = SiglipImageProcessor.from_pretrained(
            path, local_files_only=local_files_only
        )
        tokenizer = SiglipTokenizer.from_pretrained(path, local_files_only=local_files_only)
        self.processor = SiglipProcessor(image_processor=image_processor, tokenizer=tokenizer)
        self.model = AutoModel.from_pretrained(
            path,
            local_files_only=local_files_only,
            torch_dtype=torch.bfloat16,
        ).to(self.device).eval()

    def encode_images(
        self, images: Sequence[np.ndarray], batch_size: int = 24
    ) -> np.ndarray:
        outputs = []
        for start in range(0, len(images), batch_size):
            inputs = self.processor(
                images=list(images[start : start + batch_size]), return_tensors="pt"
            )
            pixels = inputs["pixel_values"].to(self.device, dtype=self.torch.bfloat16)
            with self.torch.inference_mode():
                features = self.model.get_image_features(pixel_values=pixels)
                features = self.torch.nn.functional.normalize(features.float(), dim=-1)
            outputs.append(features.cpu().numpy().astype(np.float16))
        return np.concatenate(outputs, axis=0)

    def encode_texts(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        outputs = []
        for start in range(0, len(texts), batch_size):
            inputs = self.processor(
                text=list(texts[start : start + batch_size]),
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            with self.torch.inference_mode():
                features = self.model.get_text_features(**inputs)
                features = self.torch.nn.functional.normalize(features.float(), dim=-1)
            outputs.append(features.cpu().numpy().astype(np.float16))
        return np.concatenate(outputs, axis=0)

