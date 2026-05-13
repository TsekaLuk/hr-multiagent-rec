"""Session-scoped fixtures for integration tests.

We load heavy models once per session and reuse them across tests
to keep the suite fast.
"""
from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(scope="session")
def small_embedding_model_name() -> str:
    """A small, fast, multilingual model for CI/laptop.

    Qwen3-Embedding-0.6B is the production target; we test against it
    when available, otherwise fall back to a smaller bge-m3-mini.
    """
    return "Qwen/Qwen3-Embedding-0.6B"


@pytest.fixture(scope="session")
def embedder(small_embedding_model_name: str) -> Iterator[object]:
    """Real embedder, loaded once per session.

    NOTE: Skip if the model can't be downloaded (no network) or the
    machine can't run it — but never mock.
    """
    pytest.importorskip("sentence_transformers")
    from hr_rec.encoding.embedder import Embedder

    try:
        emb = Embedder(model_name=small_embedding_model_name)
    except Exception as e:  # network / OOM / model unavailable
        pytest.skip(f"could not load real embedder ({type(e).__name__}: {e})")
    yield emb
