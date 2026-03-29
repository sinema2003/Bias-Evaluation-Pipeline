import re
from typing import Any, Dict, List

REAL_TOXICITY_PROMPTS_ID = "allenai/real-toxicity-prompts"
CROWS_PAIRS_ID = "crows_pairs"
CROWS_PAIRS_CSV_URL = (
    "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/"
    "crows_pairs_anonymized.csv"
)


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


def _safe_crows_prompt_id(idx: int, bias_type: Any, stereo: Any, anti: Any) -> str:
    base = f"{idx}_{bias_type}_{stereo}_{anti}"
    return re.sub(r"[^\w.\-]+", "_", str(base))[:200]


def _first_nonempty(row: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        val = row.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return ""


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


def load_crows_pairs_prompts(
    max_items: int,
    seed: int = 42,
    split: str = "test",
) -> List[Dict[str, Any]]:
    """
    Load prompts from CrowS-Pairs (Nangia et al., 2020).
    Returns list-of-dicts compatible with read_jsonl(...) in io_utils.
    """
    from datasets import load_dataset

    try:
        ds = load_dataset(CROWS_PAIRS_ID, split=split)
    except RuntimeError as exc:
        # Newer `datasets` versions disable legacy dataset scripts.
        # CrowS-Pairs on HF is script-only, so fall back to loading the canonical CSV.
        if "Dataset scripts are no longer supported" not in str(exc):
            raise
        if split != "test":
            raise ValueError(
                f"CrowS-Pairs only provides a test split when loaded from CSV; got split={split!r}"
            ) from exc
        ds = load_dataset("csv", data_files={"test": CROWS_PAIRS_CSV_URL}, split="test")

    ds = ds.shuffle(seed=seed)
    if max_items > 0:
        ds = ds.select(range(min(max_items, len(ds))))

    items: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        stereotype = _first_nonempty(row, ["sent_more", "stereo", "sent_stereo"])
        anti_stereotype = _first_nonempty(row, ["sent_less", "anti_stereo", "sent_antistereo"])
        if not stereotype or not anti_stereotype:
            continue

        bias_type = _first_nonempty(row, ["bias_type", "bias"]) or "unknown"
        pid = _safe_crows_prompt_id(idx, bias_type, stereotype[:32], anti_stereotype[:32])
        items.append({
            "id": f"crows_{pid}",
            "category": f"crows_pairs_{bias_type.lower()}",
            "variants": [
                {"attribute": "stereotype", "prompt": stereotype},
                {"attribute": "anti_stereotype", "prompt": anti_stereotype},
            ],
        })
    return items
