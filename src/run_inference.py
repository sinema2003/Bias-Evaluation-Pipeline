import os
import time
import json
import logging
import argparse
from typing import Dict, Any, List
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.io_utils import read_jsonl, write_jsonl
from src.config import default_decoding_configs
from src.prompt_sources import (
    REAL_TOXICITY_PROMPTS_ID,
    CROWS_PAIRS_ID,
    load_real_toxicity_prompts,
    load_crows_pairs_prompts,
)


def _device():
    return "cpu"

def generate_one(model, tok, prompt: str, gen_params: Dict[str, Any]) -> str:
    log = logging.getLogger(__name__)
    t0 = time.perf_counter()
    inp = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inp,
            pad_token_id=tok.eos_token_id,
            **gen_params,
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    dt = time.perf_counter() - t0
    log.debug(
        "generate_one done in %.3fs (prompt_len=%d chars, gen_params=%s)",
        dt,
        len(prompt),
        gen_params,
    )
    return text[len(prompt):].strip() if text.startswith(prompt) else text.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--prompt-source",
        choices=["jsonl", "rtp", "crows"],
        default="rtp",
        help=(
            "rtp = Allen AI Real Toxicity Prompts (Gehman et al., 2020); "
            "crows = CrowS-Pairs (Nangia et al., 2020); "
            "jsonl = local file."
        ),
    )
    ap.add_argument(
        "--prompts",
        default="data/prompts/prompts.jsonl",
        help="Path to JSONL when --prompt-source=jsonl.",
    )
    ap.add_argument(
        "--rtp-max-prompts",
        type=int,
        default=30,
        help="How many RTP prompts to use (streaming subset; all are unique spans).",
    )
    ap.add_argument(
        "--rtp-seed",
        type=int,
        default=42,
        help="Shuffle seed for the streaming RTP subset.",
    )
    ap.add_argument(
        "--crows-max-prompts",
        type=int,
        default=30,
        help="How many CrowS-Pairs rows to use.",
    )
    ap.add_argument(
        "--crows-seed",
        type=int,
        default=42,
        help="Shuffle seed for CrowS-Pairs.",
    )
    ap.add_argument(
        "--crows-split",
        default="test",
        help="Dataset split for CrowS-Pairs.",
    )
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--out_dir", default="outputs/runs")
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (use DEBUG to trace each generation).",
    )
    args = ap.parse_args()

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir, f"{run_id}.jsonl")
    os.makedirs(args.out_dir, exist_ok=True)

    log.info(
        "Starting run_id=%s model=%s prompt_source=%s samples=%d out=%s",
        run_id,
        args.model,
        args.prompt_source,
        args.samples,
        out_path,
    )

    t_read = time.perf_counter()
    if args.prompt_source == "jsonl":
        prompts = read_jsonl(args.prompts)
        prompt_ref = args.prompts
    elif args.prompt_source == "rtp":
        prompts = load_real_toxicity_prompts(
            max_items=args.rtp_max_prompts,
            seed=args.rtp_seed,
        )
        prompt_ref = (
            f"{REAL_TOXICITY_PROMPTS_ID} max={args.rtp_max_prompts} seed={args.rtp_seed}"
        )
    else:
        prompts = load_crows_pairs_prompts(
            max_items=args.crows_max_prompts,
            seed=args.crows_seed,
            split=args.crows_split,
        )
        prompt_ref = (
            f"{CROWS_PAIRS_ID} split={args.crows_split} "
            f"max={args.crows_max_prompts} seed={args.crows_seed}"
        )
    log.info(
        "Loaded %d prompt row(s) in %.2fs (%s)",
        len(prompts),
        time.perf_counter() - t_read,
        prompt_ref,
    )

    decs = default_decoding_configs()
    n_variants = sum(len(p["variants"]) for p in prompts)
    total = n_variants * len(decs) * args.samples
    log.info(
        "Work queue: %d variants x %d decoding configs x %d samples = %d generations",
        n_variants,
        len(decs),
        args.samples,
        total,
    )

    device = _device()
    log.info("Loading tokenizer: %s", args.model)
    t_tok = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(args.model)
    log.info("Tokenizer loaded in %.2fs", time.perf_counter() - t_tok)

    log.info("Loading model weights (this can take a long time on first download)...")
    t_m = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.model)
    log.info("Model from_pretrained finished in %.2fs", time.perf_counter() - t_m)

    t_dev = time.perf_counter()
    model.to(device)
    model.eval()
    log.info("Model on device=%s, eval() in %.2fs", device, time.perf_counter() - t_dev)

    rows: List[Dict[str, Any]] = []

    pbar = tqdm(total=total, desc="Generating")
    for item in prompts:
        pid = item["id"]
        cat = item["category"]
        log.debug("Prompt item id=%r category=%r variants=%d", pid, cat, len(item["variants"]))
        for v in item["variants"]:
            attr = v["attribute"]
            prompt = v["prompt"]
            log.debug("Variant attribute=%r prompt_preview=%r", attr, prompt[:120] + ("..." if len(prompt) > 120 else ""))
            for dc in decs:
                log.debug("Decoding=%s (about to run %d samples)", dc.name, args.samples)
                for s in range(args.samples):
                    gen = generate_one(model, tok, prompt, dc.params)
                    rows.append({
                        "run_id": run_id,
                        "model": args.model,
                        "decoding": dc.name,
                        "sample": s,
                        "prompt_id": pid,
                        "category": cat,
                        "attribute": attr,
                        "prompt": prompt,
                        "generation": gen,
                    })
                    pbar.update(1)
    pbar.close()

    log.info("Writing %d rows to %s", len(rows), out_path)
    t_w = time.perf_counter()
    write_jsonl(out_path, rows)
    log.info("write_jsonl done in %.2fs", time.perf_counter() - t_w)

    meta = {
        "run_id": run_id,
        "model": args.model,
        "samples": args.samples,
        "prompt_source": args.prompt_source,
        "prompt_file": (
            args.prompts
            if args.prompt_source == "jsonl"
            else (REAL_TOXICITY_PROMPTS_ID if args.prompt_source == "rtp" else CROWS_PAIRS_ID)
        ),
        "rtp_max_prompts": args.rtp_max_prompts if args.prompt_source == "rtp" else None,
        "rtp_seed": args.rtp_seed if args.prompt_source == "rtp" else None,
        "crows_max_prompts": args.crows_max_prompts if args.prompt_source == "crows" else None,
        "crows_seed": args.crows_seed if args.prompt_source == "crows" else None,
        "crows_split": args.crows_split if args.prompt_source == "crows" else None,
        "decodings": [d.name for d in decs],
        "device": _device(),
    }
    meta_path = os.path.join(args.out_dir, f"{run_id}.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log.info("Saved outputs: %s and %s", out_path, meta_path)

if __name__ == "__main__":
    main()