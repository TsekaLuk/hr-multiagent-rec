"""Cross-encoder reranker for Qwen3-Reranker.

Qwen3-Reranker is a *causal-LM* style reranker (not a regression
head). It judges relevance by scoring the next-token probability of
``"yes"`` vs ``"no"`` given a templated prompt. We compute the score as
``sigmoid(logit_yes - logit_no)``.

References: HuggingFace card https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_model_path(model_name: str) -> str:
    """Prefer ModelScope cache when present (CN-friendly), same pattern as Embedder."""
    import os

    ms_root = Path(os.environ.get("MODELSCOPE_CACHE", Path.home() / ".cache" / "modelscope"))
    ms_path = ms_root / "hub" / "models" / model_name.replace(".", "___")
    if ms_path.exists() and any(ms_path.glob("*.safetensors")):
        return str(ms_path)
    return model_name


_DEFAULT_INSTRUCTION = (
    "Given a job description, retrieve the resume that best matches the requirements."
)

_PROMPT_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)

_PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Reranker:
    """Qwen3-Reranker (causal-LM yes/no).

    Parameters
    ----------
    model_name:
        HF model id or local path. e.g. ``"Qwen/Qwen3-Reranker-0.6B"``.
    device:
        ``cuda | mps | cpu | None`` (auto).
    instruction:
        Task instruction. Default is tuned for job→resume retrieval.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        *,
        device: str | None = None,
        instruction: str = _DEFAULT_INSTRUCTION,
        max_length: int = 1024,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = device or _detect_device()
        self.instruction = instruction
        self.max_length = max_length

        target = _resolve_model_path(model_name)
        logger.info("loading reranker %s on %s", target, self.device)
        self._tok = AutoTokenizer.from_pretrained(target, padding_side="left")
        # MPS prefers fp16; CPU prefers fp32 for stability.
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        self._model = (
            AutoModelForCausalLM.from_pretrained(target, torch_dtype=dtype)
            .to(self.device)
            .eval()
        )
        # Resolve yes/no token ids.
        self._yes_id = self._tok.convert_tokens_to_ids("yes")
        self._no_id = self._tok.convert_tokens_to_ids("no")
        if self._yes_id is None or self._no_id is None:
            raise RuntimeError(
                "Qwen3-Reranker tokenizer missing 'yes'/'no' tokens; check model id."
            )

    # ---- public ---------------------------------------------------------

    def score_pair(self, query: str, doc: str) -> float:
        return float(self.score_pairs([(query, doc)])[0])

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int = 4,
    ) -> np.ndarray:
        if not pairs:
            return np.zeros(0, dtype=np.float32)

        scores: list[float] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            prompts = [self._format(q, d) for q, d in batch]
            enc = self._tok(
                prompts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self._model(**enc).logits[:, -1, :]
                yes = logits[:, self._yes_id]
                no = logits[:, self._no_id]
                s = torch.sigmoid(yes - no).cpu().float().numpy()
            scores.extend(s.tolist())
        return np.asarray(scores, dtype=np.float32)

    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[str, str]],
        *,
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []
        pairs = [(query, doc) for _, doc in candidates]
        scores = self.score_pairs(pairs)
        order = np.argsort(-scores)
        ranked = [(candidates[i][0], float(scores[i])) for i in order]
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked

    # ---- helpers --------------------------------------------------------

    def _format(self, query: str, doc: str) -> str:
        return (
            _PROMPT_PREFIX
            + f"<Instruct>: {self.instruction}\n"
            + f"<Query>: {(query or '').strip()}\n"
            + f"<Document>: {(doc or '').strip()}"
            + _PROMPT_SUFFIX
        )
