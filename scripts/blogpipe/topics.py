from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Track:
    name: str
    keywords: tuple[str, ...]
    required_any: tuple[str, ...] = ()
    label: str = ""


# Lower rank number = higher editorial priority for daily posts.
TRACK_PRIORITY: tuple[str, ...] = (
    "ml_engineering",
    "applied_research",
    "ml_theory",
    "aec",
    "popular_ml",
)

TRACK_WEIGHTS: dict[str, float] = {
    "ml_engineering": 1.00,
    "applied_research": 0.85,
    "ml_theory": 0.70,
    "aec": 0.55,
    "popular_ml": 0.40,
}

TRACK_LABELS: dict[str, str] = {
    "ml_engineering": "ML-Engineering",
    "applied_research": "Applied-Research",
    "ml_theory": "ML-Theory",
    "aec": "AEC",
    "popular_ml": "Popular-ML",
}

THEORY_INTEREST_CUES: tuple[str, ...] = (
    "theorem", "theoretical", "proof", "surprising", "breakthrough", "novel result",
    "lower bound", "upper bound", "first ", "state-of-the-art", "sota",
)

TRACKS: tuple[Track, ...] = (
    Track(
        "ml_engineering",
        (
            "scaling", "distributed training", "gpu", "cuda", "kernel", "triton",
            "flash attention", "kv cache", "inference", "serving", "throughput",
            "latency", "optimization", "pytorch", "jax", "huggingface", "vllm",
            "deepspeed", "megatron", "fsdp", "tensor parallel", "pipeline parallel",
            "quantization", "memory bandwidth", "data pipeline", "training efficiency",
            "compiler", "scheduling", "profiling", "observability", "monitoring",
            "feature store", "vector database", "retrieval system", "deployment",
            "infrastructure", "reproducibility",
        ),
        label="ML-Engineering",
    ),
    Track(
        "applied_research",
        (
            "language model", "llm", "transformer", "reasoning", "alignment",
            "post-training", "post training", "long-context", "long context",
            "agent", "rag", "fine-tuning", "rlhf", "dpo", "sft", "multimodal",
            "vision-language", "benchmark", "evaluation", "dataset", "ablation",
            "architecture", "objective", "retrieval", "tool use", "instruction tuning",
        ),
        label="Applied-Research",
    ),
    Track(
        "ml_theory",
        (
            "theorem", "theoretical", "bound", "complexity", "convergence",
            "generalization", "optimality", "lower bound", "upper bound",
            "proof", "analysis", "sample complexity", "regret", "pac",
        ),
        label="ML-Theory",
    ),
    Track(
        "aec",
        (
            "bim", "ifc", "cad", "digital twin", "hvac", "construction",
            "facility", "building", "architecture engineering construction",
            "aec", "revit", "autodesk", "point cloud", "scan-to-bim",
            "structural engineering", "building controls", "drawing sheet",
            "document intelligence", "layout understanding",
        ),
        required_any=(
            "bim", "ifc", "cad", "digital twin", "hvac", "construction",
            "facility", "building controls", "revit", "aec", "scan-to-bim",
            "document intelligence",
        ),
        label="AEC",
    ),
    Track(
        "popular_ml",
        (
            "frontier model", "foundation model", "gpt", "gemini", "llama",
            "chatgpt", "open source release", "model release", "industry trend",
            "widely adopted", "viral", "headline", "announcement",
        ),
        label="Popular-ML",
    ),
)


def keyword_hits(text: str, keywords: tuple[str, ...]) -> list[str]:
    blob = (text or "").lower()
    return [kw for kw in keywords if kw in blob]


def track_score(name: str, hits: list[str], *, text: str = "") -> float:
    raw = min(1.0, len(hits) / 5.0)
    if name == "ml_theory" and raw > 0 and not keyword_hits(text, THEORY_INTEREST_CUES):
        raw *= 0.45
    return raw


def weighted_best(scores: dict[str, float]) -> float:
    if not scores:
        return 0.0
    return max(scores.get(name, 0.0) * TRACK_WEIGHTS.get(name, 0.5) for name in TRACK_PRIORITY)


def priority_track(scores: dict[str, float], *, threshold: float = 0.12) -> str:
    for name in TRACK_PRIORITY:
        if scores.get(name, 0.0) >= threshold:
            return name
    return ""


def active_track_labels(scores: dict[str, float], *, threshold: float = 0.25) -> list[str]:
    out: list[str] = []
    for name in TRACK_PRIORITY:
        if scores.get(name, 0.0) >= threshold:
            label = TRACK_LABELS.get(name, name)
            if label not in out:
                out.append(label)
    return out or ["ML"]
