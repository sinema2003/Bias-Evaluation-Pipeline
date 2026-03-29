import re
from typing import Any, Dict, List

REAL_TOXICITY_PROMPTS_ID = "allenai/real-toxicity-prompts"


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _toxicity_bin(score: float) -> str:
    if score < 0.25:
        return "tox_bin_[0,0.25)"
    if score < 0.50:
        return "tox_bin_[0.25,0.50)"
    if score < 0.75:
        return "tox_bin_[0.50,0.75)"
    return "tox_bin_[0.75,1.0]"


def _prompt_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    p = row.get("prompt")
    if isinstance(p, dict):
        return p
    raise ValueError("Unexpected Real Toxicity Prompts row: missing prompt dict")


def _safe_prompt_id(filename: str, begin: Any, end: Any) -> str:
    base = f"{filename}_{begin}_{end}"
    return re.sub(r"[^\w.\-]+", "_", str(base))[:200]


def load_real_toxicity_prompts(
    max_items: int,
    seed: int = 42,
    buffer_size: int = 10_000,
) -> List[Dict[str, Any]]:
    """
    Load prompts from Allen AI Real Toxicity Prompts (Gehman et al., 2020).
    Returns the same list-of-dicts shape as read_jsonl(...) in io_utils.
    """
    from datasets import load_dataset

    ds = load_dataset(REAL_TOXICITY_PROMPTS_ID, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=buffer_size)

    items: List[Dict[str, Any]] = []
    for row in ds:
        if len(items) >= max_items:
            break
        payload = _prompt_payload(row)
        text = payload.get("text")
        if not text or not str(text).strip():
            continue
        tox = _as_float(payload.get("toxicity"), 0.0)
        challenging = bool(row.get("challenging", False))
        filename = row.get("filename", "unknown")
        begin = row.get("begin", 0)
        end = row.get("end", 0)
        pid = _safe_prompt_id(filename, begin, end)
        attribute = f"{_toxicity_bin(tox)}|challenging={challenging}"
        items.append({
            "id": f"rtp_{pid}",
            "category": "real_toxicity_prompts",
            "variants": [{"attribute": attribute, "prompt": str(text).strip()}],
        })
    return items
