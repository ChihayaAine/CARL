"""Episode logger.

One :class:`EpisodeRecord` per (x, T, Y) execution; persisted to JSONL files
under ``runs/<task>/logs/``. The record format is the single source of truth
for everything downstream: nuisance training, DR, calibration, baselines,
analysis, and table generation.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


@dataclass
class EpisodeRecord:
    example_id: str
    task: str
    benchmark_id: str
    treatment_id: str
    treatment_index: int
    treatment_hash: str
    # behavior policy: full distribution over protocols + sampled propensity
    propensity: float                      # b(T_i | X_i)
    propensity_vector: List[float]         # b(. | X_i)
    behavior_source: str                   # 'probe' | 'exploit' | 'behavior'
    # outcomes
    reward: float                          # R(k) in [0, 1]
    token_cost_norm: float                 # C_tok(k) / ref_tokens
    latency_cost_norm: float               # C_lat(k) / ref_latency
    utility: float                         # Y(k) = R - lambda*C_tok - mu*C_lat
    # raw counts
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    # auxiliary
    prediction: str
    difficulty_hint: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


class EpisodeLogger:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a")

    def __enter__(self) -> "EpisodeLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def write(self, record: EpisodeRecord) -> None:
        self._fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


def write_episodes(path: Path | str, records: Iterable[EpisodeRecord]) -> None:
    with EpisodeLogger(path) as logger:
        for r in records:
            logger.write(r)


def read_episodes(path: Path | str) -> List[EpisodeRecord]:
    out: List[EpisodeRecord] = []
    p = Path(path)
    if not p.exists():
        return out
    with open(p, "r") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            out.append(EpisodeRecord(**d))
    return out


def stream_episodes(path: Path | str) -> Iterator[EpisodeRecord]:
    p = Path(path)
    if not p.exists():
        return
    with open(p, "r") as f:
        for line in f:
            if line.strip():
                yield EpisodeRecord(**json.loads(line))
