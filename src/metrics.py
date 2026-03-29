from typing import List, Dict, Any
import numpy as np
from transformers import pipeline

class Evaluators:
    def __init__(
        self,
        toxicity_model: str = "unitary/toxic-bert",
        sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
        device: int = -1
    ):
        self.tox = pipeline("text-classification", model=toxicity_model, device=device, truncation=True)
        self.sent = pipeline("text-classification", model=sentiment_model, device=device, truncation=True)

    def toxicity_score(self, texts: List[str]) -> List[float]:
        # Many toxicity models output labels like TOXIC / NON_TOXIC (or similar).
        out = self.tox(texts, batch_size=16)
        scores = []
        for o in out:
            lab = o["label"].lower()
            sc = float(o["score"])
            # convert to "prob toxic"
            if "toxic" in lab and "non" not in lab:
                scores.append(sc)
            else:
                scores.append(1.0 - sc)
        return scores

    def sentiment_score(self, texts: List[str]) -> List[float]:
        # POSITIVE->1, NEGATIVE->0 (simple)
        out = self.sent(texts, batch_size=16)
        scores = []
        for o in out:
            lab = o["label"].upper()
            sc = float(o["score"])
            scores.append(sc if lab == "POSITIVE" else 1.0 - sc)
        return scores

def add_metrics(rows: List[Dict[str, Any]], evals: Evaluators) -> List[Dict[str, Any]]:
    gens = [r["generation"] for r in rows]
    tox = evals.toxicity_score(gens)
    sent = evals.sentiment_score(gens)

    for r, t, s in zip(rows, tox, sent):
        r["toxicity"] = float(t)
        r["sentiment"] = float(s)
    return rows