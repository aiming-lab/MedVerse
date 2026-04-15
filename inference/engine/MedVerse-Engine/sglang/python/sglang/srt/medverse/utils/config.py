"""Global configuration for the MedVerse inference engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from sglang.srt.medverse.utils.constants import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    MAX_BATCH_SIZE,
    MAX_DAG_DEPTH,
    MAX_PARALLEL_BRANCHES,
)


@dataclass
class DAGConfig:
    """Per-request DAG execution configuration."""

    max_depth: int = MAX_DAG_DEPTH
    max_parallel_branches: int = MAX_PARALLEL_BRANCHES
    merge_strategy: Literal["evidence_weighted", "concat", "best_path"] = "evidence_weighted"
    enable_pruning: bool = True
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD


@dataclass
class EngineConfig:
    """Global engine-wide configuration."""

    model_path: str = ""
    tp_size: int = 1
    max_batch_size: int = MAX_BATCH_SIZE
    max_total_tokens: int = 131072
    mem_fraction_static: float = 0.9
    attention_backend: Literal["flashinfer", "triton"] = "flashinfer"
    dtype: str = "auto"
    dag: DAGConfig = field(default_factory=DAGConfig)
    host: str = "127.0.0.1"
    port: int = 30000
    tokenizer_path: str | None = None
    trust_remote_code: bool = False
    schedule_policy: Literal["lpm", "fcfs", "dfs_weight", "lof", "random", "dag_priority"] = (
        "dag_priority"
    )
    section_cache_size: int = 1024
