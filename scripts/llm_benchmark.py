"""Compare LLM TTFT/total latency across candidates using the real clinic
system prompt, streaming, multiple runs each. Read-only - does not change
agent.py's configured provider. Run this, look at the numbers, then decide.

Usage: python scripts/llm_benchmark.py [--runs 6]
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

# Realistic clinic turn: system prompt matches agent.py's build_initial_context
# output (extracted live, ~1180 tokens), user turn matches a normal mid-call
# response.
SYSTEM_PROMPT = (ROOT / "scratch_system_prompt.txt").read_text(encoding="utf-8")
USER_TURN = "I have PCOS and want to book with a doctor tomorrow evening, my name is Priya."


def bench_openai_compatible(label: str, base_url: str, api_key: str, model: str) -> dict:
    started = time.perf_counter()
    ttft_ms = None
    text = ""
    try:
        with requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_TURN},
                ],
                "max_tokens": 90,
                "temperature": 0.25,
                "stream": True,
            },
            stream=True,
            timeout=30,
        ) as resp:
            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="ignore")
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    delta = chunk["choices"][0]["delta"].get("content")
                except (KeyError, IndexError, ValueError):
                    continue
                if delta:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - started) * 1000
                    text += delta
        total_ms = (time.perf_counter() - started) * 1000
        return {"ttft_ms": round(ttft_ms, 1) if ttft_ms else None, "total_ms": round(total_ms, 1), "chars": len(text)}
    except Exception as e:
        return {"error": str(e)}


def bench_gemini(model: str) -> dict:
    key = ENV.get("GEMINI_API_KEY")
    if not key:
        return {"error": "GEMINI_API_KEY not configured"}
    started = time.perf_counter()
    ttft_ms = None
    text = ""
    try:
        with requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent",
            params={"alt": "sse", "key": key},
            headers={"Content-Type": "application/json"},
            json={
                "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": [{"text": USER_TURN}]}],
                "generationConfig": {"maxOutputTokens": 90, "temperature": 0.25},
            },
            stream=True,
            timeout=30,
        ) as resp:
            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="ignore")
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload:
                    continue
                try:
                    chunk = json.loads(payload)
                    parts = chunk["candidates"][0]["content"]["parts"]
                    delta = "".join(p.get("text", "") for p in parts)
                except (KeyError, IndexError, ValueError):
                    continue
                if delta:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - started) * 1000
                    text += delta
        total_ms = (time.perf_counter() - started) * 1000
        return {"ttft_ms": round(ttft_ms, 1) if ttft_ms else None, "total_ms": round(total_ms, 1), "chars": len(text)}
    except Exception as e:
        return {"error": str(e)}


CANDIDATES = {
    "openai/gpt-4o-mini (current primary)": lambda: bench_openai_compatible(
        "openai", "https://api.openai.com/v1", ENV.get("OPENAI_API_KEY", ""), "gpt-4o-mini"
    ),
    "groq/llama-3.1-8b-instant (current fallback)": lambda: bench_openai_compatible(
        "groq", "https://api.groq.com/openai/v1", ENV.get("GROQ_API_KEY", ""), "llama-3.1-8b-instant"
    ),
    "groq/qwen3-32b (candidate)": lambda: bench_openai_compatible(
        "groq", "https://api.groq.com/openai/v1", ENV.get("GROQ_API_KEY", ""), "qwen/qwen3-32b"
    ),
    "groq/allam-2-7b (smallest available)": lambda: bench_openai_compatible(
        "groq", "https://api.groq.com/openai/v1", ENV.get("GROQ_API_KEY", ""), "allam-2-7b"
    ),
    "groq/openai-gpt-oss-20b (currently configured)": lambda: bench_openai_compatible(
        "groq", "https://api.groq.com/openai/v1", ENV.get("GROQ_API_KEY", ""), "openai/gpt-oss-20b"
    ),
    "gemini-2.5-flash (candidate)": lambda: bench_gemini("gemini-2.5-flash"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=6)
    args = parser.parse_args()

    results = {}
    for name, fn in CANDIDATES.items():
        samples = []
        errors = []
        for i in range(args.runs):
            r = fn()
            if "error" in r:
                errors.append(r["error"])
                print(f"[{name}] run {i+1}/{args.runs}: ERROR {r['error'][:100]}")
            else:
                samples.append(r)
                print(f"[{name}] run {i+1}/{args.runs}: ttft={r['ttft_ms']}ms total={r['total_ms']}ms chars={r['chars']}")
            time.sleep(0.3)
        results[name] = {"samples": samples, "errors": errors}

    print("\n" + "=" * 90)
    print(f"{'Model':<48}{'n':>4}{'TTFT p50':>12}{'TTFT p95':>12}{'Total p50':>12}")
    print("-" * 90)
    ranked = []
    for name, data in results.items():
        samples = data["samples"]
        if not samples:
            print(f"{name:<48}{'0':>4}  FAILED: {data['errors'][0][:60] if data['errors'] else 'unknown'}")
            continue
        ttfts = sorted(s["ttft_ms"] for s in samples if s["ttft_ms"] is not None)
        totals = sorted(s["total_ms"] for s in samples)
        if not ttfts:
            continue
        p50 = ttfts[len(ttfts) // 2]
        p95 = ttfts[min(len(ttfts) - 1, int(len(ttfts) * 0.95))]
        total_p50 = totals[len(totals) // 2]
        ranked.append((p50, name, p95, total_p50, len(samples)))
        print(f"{name:<48}{len(samples):>4}{p50:>10.0f}ms{p95:>10.0f}ms{total_p50:>10.0f}ms")

    ranked.sort()
    print("\nRanked by TTFT p50 (fastest first):")
    for i, (p50, name, p95, total_p50, n) in enumerate(ranked, 1):
        print(f"  {i}. {name}: {p50:.0f}ms")

    if ranked:
        print(f"\nFastest: {ranked[0][1]} ({ranked[0][0]:.0f}ms TTFT p50)")


if __name__ == "__main__":
    main()
