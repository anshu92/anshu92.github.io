from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Track:
    name: str
    keywords: tuple[str, ...]
    required_any: tuple[str, ...] = ()


TRACKS = (
    Track(
        "llm",
        (
            "llm", "language model", "transformer", "reasoning", "alignment",
            "post-training", "post training", "long-context", "long context",
            "agent", "rag", "inference", "kv cache", "token", "attention",
            "fine-tuning", "rlhf", "dpo", "sft", "multimodal",
        ),
    ),
    Track(
        "mle",
        (
            "serving", "evaluation", "observability", "data pipeline",
            "distributed training", "gpu", "throughput", "latency", "benchmark",
            "deployment", "reproducibility", "monitoring", "infrastructure",
            "feature store", "vector database", "retrieval", "kernel",
        ),
    ),
    Track(
        "aec",
        (
            "bim", "ifc", "cad", "digital twin", "hvac", "construction",
            "facility", "building", "architecture engineering construction",
            "aec", "revit", "autodesk", "point cloud", "scan-to-bim",
            "structural engineering", "building controls",
        ),
        required_any=(
            "bim", "ifc", "cad", "digital twin", "hvac", "construction",
            "facility", "building controls", "revit", "aec",
        ),
    ),
)


def keyword_hits(text: str, keywords: tuple[str, ...]) -> list[str]:
    blob = (text or "").lower()
    return [kw for kw in keywords if kw in blob]
