"""Offline latency percentile report over logs/pipeline_events.jsonl.

Consumes the per-turn `turn_latency` summary events emitted by agent.py's
TurnLatencyAggregator (plus the raw eou/llm/tts metric events for runs
recorded before the aggregator existed) and prints p50/p75/p95 per metric.

Usage:
    python scripts/latency_report.py [path/to/pipeline_events.jsonl]
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "logs" / "pipeline_events.jsonl"

METRICS = [
    ("stt_final_ms", "STT finalization"),
    ("eou_delay_ms", "End-of-utterance delay"),
    ("llm_ttft_ms", "LLM time to first token"),
    ("llm_total_ms", "LLM total"),
    ("tts_ttfa_ms", "TTS time to first audio"),
    ("first_audio_total_ms", "End of speech -> first audio (est.)"),
]


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = max(0, min(len(values) - 1, round((p / 100) * (len(values) - 1))))
    return values[k]


def main() -> None:
    if not LOG_PATH.exists():
        raise SystemExit(f"log not found: {LOG_PATH}")

    turns: list[dict] = []
    fallback_samples: dict[str, list[float]] = {"eou_delay_ms": [], "llm_ttft_ms": [], "tts_ttfa_ms": []}
    cancelled = 0
    fast_path = 0
    llm_calls = 0

    for line in LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        details = event.get("details", {}) or {}
        kind = details.get("event")
        if kind == "turn_latency":
            turns.append(details)
            if details.get("cancelled_generation"):
                cancelled += 1
            if details.get("response_path") == "llm":
                llm_calls += 1
        elif kind == "fast_path":
            fast_path += 1
        # Raw per-metric events, for logs predating the aggregator.
        elif kind == "eou_delay_ms":
            fallback_samples["eou_delay_ms"].append(float(details.get("value", 0)))
        elif kind == "llm_ttft_ms":
            fallback_samples["llm_ttft_ms"].append(float(details.get("ttft_ms", 0)))
        elif kind == "ttfa_ms":
            fallback_samples["tts_ttfa_ms"].append(float(details.get("ttfa_ms", 0)))

    print(f"log: {LOG_PATH}")
    print(f"turn_latency events: {len(turns)} | fast-path turns: {fast_path} | "
          f"llm turns: {llm_calls} | cancelled generations: {cancelled}\n")

    print(f"{'metric':<38}{'n':>5}{'p50':>10}{'p75':>10}{'p95':>10}")
    print("-" * 73)
    for key, label in METRICS:
        values = [float(t[key]) for t in turns if t.get(key) is not None]
        if not values and key in fallback_samples:
            values = fallback_samples[key]
            label += " (raw)"
        if not values:
            continue
        print(f"{label:<38}{len(values):>5}{pct(values, 50):>10.0f}{pct(values, 75):>10.0f}{pct(values, 95):>10.0f}")

    if not turns and not any(fallback_samples.values()):
        print("no latency samples found - run a call first")


if __name__ == "__main__":
    main()
