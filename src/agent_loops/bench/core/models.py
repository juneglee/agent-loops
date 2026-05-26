from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    filename: str
    quantization: str = "q4_0"


DEFAULT = "gemma-4-E4B-qat"

MODELS: dict[str, ModelSpec] = {
    "gemma-4-E4B-qat": ModelSpec(
        repo="google/gemma-4-E4B-it-qat-q4_0-gguf",
        filename="gemma-4-E4B_q4_0-it.gguf",
    ),
}
