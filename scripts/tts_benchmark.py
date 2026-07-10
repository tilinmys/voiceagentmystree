"""Benchmark curated Indian TTS voices across Sarvam / Rumik / Smallest.ai
to find the fastest + most reliable option for the MyStree Clinic agent.

Supersedes the old Sarvam-vs-60db benchmark - 60db was removed from the
runtime TTS chain (see CHANGELOG). Hits each provider's HTTP API directly (no
LiveKit room needed). Measures:

  - ttfb_ms: time from request sent to first audio byte received (true
    time-to-first-byte for Rumik and Smallest.ai, which both stream over
    HTTP). Sarvam's REST endpoint returns one buffered JSON blob (no
    streaming), so its number is full round-trip latency, not TTFB - the
    production path (sarvam_wrappers.py) uses a websocket and is materially
    faster than this REST call. Treat the Sarvam column as a ceiling, not the
    real number.
  - total_ms: time to the last byte of audio.

Usage:
    python scripts/tts_benchmark.py [--runs 3] [--out logs/tts_benchmark_report.md]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import voice_catalog  # noqa: E402

TEST_PHRASE = (
    "Good morning, thank you for calling MyStree Clinic in Indiranagar. "
    "I can help you book an appointment with our gynaecologist. "
    "Could you please tell me your name and preferred date?"
)


def load_env() -> dict:
    env = {}
    for candidate in [ROOT / ".env", ROOT.parent / ".env"]:
        if not candidate.exists():
            continue
        for raw in candidate.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    return env


ENV = load_env()


def bench_sarvam(voice_id: str) -> dict:
    key = ENV.get("SARVAM_API_KEY")
    if not key:
        return {"error": "SARVAM_API_KEY not configured"}
    started = time.perf_counter()
    try:
        resp = requests.post(
            "https://api.sarvam.ai/text-to-speech",
            headers={"api-subscription-key": key, "Content-Type": "application/json"},
            json={
                "text": TEST_PHRASE,
                "target_language_code": "en-IN",
                "speaker": voice_id,
                "model": "bulbul:v3",
            },
            timeout=30,
        )
        total_ms = (time.perf_counter() - started) * 1000
        if resp.status_code != 200:
            return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        payload = resp.json()
        audio_b64 = (payload.get("audios") or [""])[0]
        return {"ttfb_ms": None, "total_ms": round(total_ms, 1), "audio_bytes": len(audio_b64)}
    except Exception as e:
        return {"error": str(e)}


def bench_rumik(voice_id: str) -> dict:
    key = ENV.get("RUMIK_API_KEY")
    if not key:
        return {"error": "RUMIK_API_KEY not configured"}
    voice_meta = voice_catalog.RUMIK_VOICES.get(voice_id, {})
    description = voice_meta.get("description")
    if not description:
        return {"error": f"no description configured for rumik voice '{voice_id}'"}
    started = time.perf_counter()
    ttfb_ms = None
    total_bytes = 0
    try:
        with requests.post(
            "https://silk-api.rumik.ai/v1/tts",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": voice_catalog.RUMIK_DEFAULT_MODEL,
                "text": TEST_PHRASE,
                "description": description,
                "temperature": 0.4,
            },
            stream=True,
            timeout=30,
        ) as resp:
            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
            for chunk in resp.iter_content(chunk_size=2048):
                if not chunk:
                    continue
                if ttfb_ms is None:
                    ttfb_ms = (time.perf_counter() - started) * 1000
                total_bytes += len(chunk)
        total_ms = (time.perf_counter() - started) * 1000
        return {"ttfb_ms": round(ttfb_ms, 1) if ttfb_ms else None, "total_ms": round(total_ms, 1), "audio_bytes": total_bytes}
    except Exception as e:
        return {"error": str(e)}


def bench_smallest(voice_id: str) -> dict:
    key = ENV.get("SMALLEST_API_KEY")
    if not key:
        return {"error": "SMALLEST_API_KEY not configured"}
    started = time.perf_counter()
    ttfb_ms = None
    total_bytes = 0
    try:
        with requests.post(
            f"https://waves-api.smallest.ai/api/v1/{voice_catalog.SMALLEST_DEFAULT_MODEL}/get_speech",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "text": TEST_PHRASE,
                "voice_id": voice_id,
                "sample_rate": voice_catalog.SMALLEST_SAMPLE_RATE,
            },
            stream=True,
            timeout=30,
        ) as resp:
            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
            for chunk in resp.iter_content(chunk_size=2048):
                if not chunk:
                    continue
                if ttfb_ms is None:
                    ttfb_ms = (time.perf_counter() - started) * 1000
                total_bytes += len(chunk)
        total_ms = (time.perf_counter() - started) * 1000
        return {"ttfb_ms": round(ttfb_ms, 1) if ttfb_ms else None, "total_ms": round(total_ms, 1), "audio_bytes": total_bytes}
    except Exception as e:
        return {"error": str(e)}


BENCHERS = {"sarvam": bench_sarvam, "rumik": bench_rumik, "smallest": bench_smallest}


def run_benchmark(runs: int) -> list[dict]:
    results = []
    for provider, voices in voice_catalog.CATALOG.items():
        bencher = BENCHERS[provider]
        for voice_id, meta in voices.items():
            samples = []
            errors = []
            for _ in range(runs):
                result = bencher(voice_id)
                if "error" in result:
                    errors.append(result["error"])
                else:
                    samples.append(result)
                time.sleep(0.2)  # be polite to rate limits between calls
            print(f"[{provider}] {meta['name']} ({voice_id}): "
                  f"{len(samples)}/{runs} ok" + (f", errors={errors[:1]}" if errors else ""))
            if samples:
                total_median = statistics.median(s["total_ms"] for s in samples)
                ttfb_values = [s["ttfb_ms"] for s in samples if s.get("ttfb_ms") is not None]
                ttfb_median = statistics.median(ttfb_values) if ttfb_values else None
                results.append({
                    "provider": provider,
                    "voice_id": voice_id,
                    "name": meta["name"],
                    "gender": meta.get("gender"),
                    "style": meta.get("style"),
                    "ttfb_ms": round(ttfb_median, 1) if ttfb_median else None,
                    "total_ms": round(total_median, 1),
                    "samples": len(samples),
                    "errors": len(errors),
                })
            else:
                results.append({
                    "provider": provider, "voice_id": voice_id, "name": meta["name"],
                    "gender": meta.get("gender"), "style": meta.get("style"),
                    "ttfb_ms": None, "total_ms": None, "samples": 0, "errors": len(errors),
                    "error_detail": errors[0] if errors else "unknown",
                })
    return results


def rank_key(row: dict):
    # Rank by TTFB when available (closer to real perceived latency), else total_ms.
    metric = row["ttfb_ms"] if row["ttfb_ms"] is not None else row["total_ms"]
    return (metric is None, metric if metric is not None else float("inf"))


def render_report(results: list[dict]) -> str:
    ok_rows = sorted((r for r in results if r["total_ms"] is not None), key=rank_key)
    failed_rows = [r for r in results if r["total_ms"] is None]

    lines = [
        "# MyStree Clinic - TTS voice benchmark",
        "",
        f'Test phrase: "{TEST_PHRASE}"',
        "",
        "Sarvam's number is full non-streaming REST round-trip latency (no TTFB - "
        "the production path uses a websocket and is faster). Rumik and "
        "Smallest.ai numbers are true time-to-first-audio-byte over HTTP streaming.",
        "",
        "| Rank | Provider | Voice | Gender | Style | TTFB (ms) | Total (ms) | Samples |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, row in enumerate(ok_rows, start=1):
        lines.append(
            f"| {i} | {row['provider']} | {row['name']} (`{row['voice_id']}`) | "
            f"{row['gender'] or '-'} | {row['style'] or '-'} | "
            f"{row['ttfb_ms'] if row['ttfb_ms'] is not None else '-'} | "
            f"{row['total_ms']} | {row['samples']} |"
        )

    if failed_rows:
        lines.append("")
        lines.append("## Failed / unavailable")
        lines.append("| Provider | Voice | Error |")
        lines.append("|---|---|---|")
        for row in failed_rows:
            lines.append(f"| {row['provider']} | {row['name']} (`{row['voice_id']}`) | {row.get('error_detail', 'unknown')} |")

    if ok_rows:
        best_per_provider = {}
        for row in ok_rows:
            best_per_provider.setdefault(row["provider"], row)
        lines.append("")
        lines.append("## Best voice per provider")
        for provider, row in best_per_provider.items():
            metric_label = "TTFB" if row["ttfb_ms"] is not None else "total"
            metric_val = row["ttfb_ms"] if row["ttfb_ms"] is not None else row["total_ms"]
            lines.append(f"- **{provider}**: {row['name']} (`{row['voice_id']}`) - {metric_val} ms {metric_label}")

        overall = ok_rows[0]
        lines.append("")
        lines.append(
            f"## Recommendation: fastest overall is **{overall['provider']} / {overall['name']}** "
            f"(`{overall['voice_id']}`)."
        )
        lines.append(
            "This is a latency-only signal - listen to the actual samples (voice quality, "
            "Indian-accent naturalness, prosody on clinic vocabulary) before committing. "
            "Latency differences under ~150ms will not be perceptible to a caller."
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=2, help="Samples per voice (default 2)")
    parser.add_argument("--out", type=str, default="logs/tts_benchmark_report.md")
    parser.add_argument("--json-out", type=str, default="logs/tts_benchmark_results.json")
    args = parser.parse_args()

    results = run_benchmark(args.runs)
    report = render_report(results)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    json_path = ROOT / args.json_out
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + report)
    print(f"\nReport written to {out_path}")
    print(f"Raw results written to {json_path}")


if __name__ == "__main__":
    main()
