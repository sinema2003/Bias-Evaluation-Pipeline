from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass(frozen=True)
class DecodingConfig:
    name: str
    params: Dict[str, Any]

def default_decoding_configs() -> List[DecodingConfig]:
    return [
        DecodingConfig("greedy", dict(do_sample=False, max_new_tokens=60)),
        DecodingConfig("t07_p09", dict(do_sample=True, temperature=0.7, top_p=0.9, max_new_tokens=60)),
        DecodingConfig("t11_p095", dict(do_sample=True, temperature=1.1, top_p=0.95, max_new_tokens=60)),
    ]