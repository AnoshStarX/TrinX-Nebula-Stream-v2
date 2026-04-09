#!/usr/bin/env python3
"""Phase 5 evaluation harness for latency, memory quality, and cost proxies.

Runs benchmark turns against /chat and /chat/stream, then computes:
- p50/p95/p99 latency
- stream first-token latency (TTFT)
- memory recall/precision proxy
- stale-memory error rate
- estimated cost per 1k turns (optional pricing inputs)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import httpx


@dataclass
class TurnMetric:
    endpoint: str
    prompt: str
    session_id: str
    user_id: str
    latency_ms: float
    first_token_ms: Optional[float]
    status_code: int
    error: Optional[str]
    input_chars: int
    output_chars: int


@dataclass
class GroundednessMetric:
    endpoint: str
    prompt: str
    score: float
    matched_expected: int
    expected_total: int
    forbidden_hits: int
    has_sources: bool
    status_code: int
    error: Optional[str]
    response_excerpt: str


def _percentile(values: list[float], p: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 2)
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (p / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    if low == high:
        return round(ordered[low], 2)
    interp = ordered[low] + (ordered[high] - ordered[low]) * (rank - low)
    return round(interp, 2)


def _summarize_turns(turns: list[TurnMetric]) -> dict[str, Any]:
    latencies = [t.latency_ms for t in turns if not t.error and t.status_code == 200]
    ttft = [t.first_token_ms for t in turns if t.first_token_ms is not None and not t.error and t.status_code == 200]
    errors = [t for t in turns if t.error or t.status_code != 200]
    total_input_chars = sum(t.input_chars for t in turns)
    total_output_chars = sum(t.output_chars for t in turns)
    return {
        "turns": len(turns),
        "success": len(turns) - len(errors),
        "errors": len(errors),
        "latency_ms": {
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "p99": _percentile(latencies, 99),
            "avg": round(sum(latencies) / len(latencies), 2) if latencies else None,
        },
        "first_token_ms": {
            "p50": _percentile(ttft, 50),
            "p95": _percentile(ttft, 95),
            "p99": _percentile(ttft, 99),
            "avg": round(sum(ttft) / len(ttft), 2) if ttft else None,
        },
        "chars": {
            "input": total_input_chars,
            "output": total_output_chars,
        },
    }


def _extract_source_urls(text: str) -> list[str]:
    # Keep URL extraction simple and deterministic.
    import re

    return re.findall(r"https?://[^\s\]\)]+", text or "")


def _load_groundedness_cases(path: str) -> list[dict[str, Any]]:
    default_cases = [
        {
            "name": "planet-facts",
            "prompt": (
                "Use only these facts:\n"
                "1) Mercury is the closest planet to the Sun.\n"
                "2) Venus is the hottest planet.\n"
                "Question: Which planet is closest and which is hottest?"
            ),
            "expected_terms": ["mercury", "venus"],
            "forbidden_terms": ["earth is closest", "mars is hottest"],
            "require_sources": False,
        },
        {
            "name": "capital-facts",
            "prompt": (
                "Answer using only this context:\n"
                "France capital: Paris.\n"
                "Germany capital: Berlin.\n"
                "Question: What is the capital of France and Germany?"
            ),
            "expected_terms": ["paris", "berlin"],
            "forbidden_terms": ["london", "madrid"],
            "require_sources": False,
        },
    ]
    if not path:
        return default_cases
    if not os.path.exists(path):
        return default_cases
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, list) and loaded:
            return loaded
    except Exception:
        return default_cases
    return default_cases


def _load_memory_cases(path: str) -> list[dict[str, Any]]:
    default_cases = [
        {
            "name": "favorite-overwrite",
            "setup_turns": [
                "Remember that my favorite coin is Bitcoin.",
                "Remember that my favorite coin is Ethereum.",
            ],
            "query": "What is my favorite coin?",
            "must_have": ["ethereum"],
            "should_not_have": ["bitcoin"],
        },
        {
            "name": "location-recall",
            "setup_turns": ["Remember that I live in Austin."],
            "query": "Where do I live?",
            "must_have": ["austin"],
            "should_not_have": [],
        },
        {
            "name": "format-preference",
            "setup_turns": ["Remember that I prefer short answers."],
            "query": "How should you format your answers for me?",
            "must_have": ["short"],
            "should_not_have": [],
        },
    ]
    if not path:
        return default_cases
    if not os.path.exists(path):
        return default_cases
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, list) and loaded:
            return loaded
    except Exception:
        return default_cases
    return default_cases


def _score_groundedness_response(case: dict[str, Any], response: str) -> tuple[float, int, int, int, bool]:
    expected_terms = [str(x).lower() for x in case.get("expected_terms", []) if str(x).strip()]
    forbidden_terms = [str(x).lower() for x in case.get("forbidden_terms", []) if str(x).strip()]
    require_sources = bool(case.get("require_sources", False))

    low = (response or "").lower()
    matched_expected = sum(1 for term in expected_terms if term in low)
    forbidden_hits = sum(1 for term in forbidden_terms if term in low)

    expected_score = 1.0
    if expected_terms:
        expected_score = matched_expected / len(expected_terms)

    urls = _extract_source_urls(response)
    has_sources = len(urls) > 0
    source_bonus = 0.0
    if require_sources:
        expected_score *= 0.8
        source_bonus = 0.2 if has_sources else 0.0

    penalty = min(0.7, 0.35 * forbidden_hits)
    final_score = max(0.0, min(1.0, expected_score + source_bonus - penalty))
    return final_score, matched_expected, len(expected_terms), forbidden_hits, has_sources


async def _call_chat(
    client: httpx.AsyncClient,
    prompt: str,
    session_id: str,
    user_id: str,
    timeout_s: float,
) -> tuple[TurnMetric, str]:
    started = time.perf_counter()
    payload = {"prompt": prompt, "session_id": session_id, "user_id": user_id}
    status_code = 0
    error = None
    response_text = ""
    try:
        resp = await client.post("/chat", json=payload, timeout=timeout_s)
        status_code = resp.status_code
        if resp.status_code == 200:
            body = resp.json()
            response_text = str(body.get("response", "")).strip()
        else:
            error = f"HTTP {resp.status_code}"
    except Exception as exc:  # pragma: no cover - network/runtime dependent
        error = str(exc)
    latency_ms = (time.perf_counter() - started) * 1000.0
    metric = TurnMetric(
        endpoint="/chat",
        prompt=prompt,
        session_id=session_id,
        user_id=user_id,
        latency_ms=round(latency_ms, 2),
        first_token_ms=None,
        status_code=status_code,
        error=error,
        input_chars=len(prompt),
        output_chars=len(response_text),
    )
    return metric, response_text


async def _call_stream(
    client: httpx.AsyncClient,
    prompt: str,
    session_id: str,
    user_id: str,
    timeout_s: float,
) -> tuple[TurnMetric, str]:
    started = time.perf_counter()
    payload = {"prompt": prompt, "session_id": session_id, "user_id": user_id}
    status_code = 0
    error = None
    first_token_ms: Optional[float] = None
    token_parts: list[str] = []
    try:
        async with client.stream(
            "POST",
            "/chat/stream",
            json=payload,
            headers={"Accept": "text/event-stream"},
            timeout=timeout_s,
        ) as resp:
            status_code = resp.status_code
            if resp.status_code != 200:
                error = f"HTTP {resp.status_code}"
            else:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    if first_token_ms is None:
                        first_token_ms = round((time.perf_counter() - started) * 1000.0, 2)
                    try:
                        payload_obj = json.loads(data)
                        tok = str(payload_obj.get("token", ""))
                        if tok:
                            token_parts.append(tok)
                    except json.JSONDecodeError:
                        continue
    except Exception as exc:  # pragma: no cover - network/runtime dependent
        error = str(exc)
    latency_ms = (time.perf_counter() - started) * 1000.0
    response_text = "".join(token_parts).strip()
    metric = TurnMetric(
        endpoint="/chat/stream",
        prompt=prompt,
        session_id=session_id,
        user_id=user_id,
        latency_ms=round(latency_ms, 2),
        first_token_ms=first_token_ms,
        status_code=status_code,
        error=error,
        input_chars=len(prompt),
        output_chars=len(response_text),
    )
    return metric, response_text


async def _run_turn(
    client: httpx.AsyncClient,
    endpoint: str,
    prompt: str,
    session_id: str,
    user_id: str,
    timeout_s: float,
) -> tuple[TurnMetric, str]:
    if endpoint == "/chat/stream":
        return await _call_stream(client, prompt, session_id, user_id, timeout_s)
    return await _call_chat(client, prompt, session_id, user_id, timeout_s)


async def _fetch_health(client: httpx.AsyncClient, timeout_s: float) -> Optional[dict[str, Any]]:
    try:
        resp = await client.get("/health", timeout=timeout_s)
        if resp.status_code == 200:
            return resp.json()
    except Exception:  # pragma: no cover - network/runtime dependent
        return None
    return None


def _contains_any(text: str, options: list[str]) -> bool:
    low = (text or "").lower()
    return any(opt.lower() in low for opt in options)


async def _run_latency_suite(client: httpx.AsyncClient, args: argparse.Namespace) -> list[TurnMetric]:
    prompts = [
        "Give me a concise update on Bitcoin and Ethereum.",
        "What is happening in AI today?",
        "Summarize current crypto market movement in 3 bullets.",
        "What is the latest update on Trinity Coin AI?",
    ]
    endpoints: list[str]
    if args.mode == "chat":
        endpoints = ["/chat"]
    elif args.mode == "stream":
        endpoints = ["/chat/stream"]
    else:
        endpoints = ["/chat", "/chat/stream"]

    turns: list[TurnMetric] = []
    for i in range(args.rounds):
        base_prompt = prompts[i % len(prompts)]
        prompt = f"{base_prompt} [run={i} seed={uuid.uuid4().hex[:6]}]"
        for ep in endpoints:
            session_id = f"lat-{ep.split('/')[-1]}-{i}-{uuid.uuid4().hex[:6]}"
            user_id = f"lat-user-{i % 3}"
            metric, _ = await _run_turn(client, ep, prompt, session_id, user_id, args.timeout)
            turns.append(metric)
    return turns


async def _run_memory_suite(client: httpx.AsyncClient, args: argparse.Namespace) -> tuple[list[TurnMetric], dict[str, Any]]:
    endpoint = args.memory_endpoint
    cases = _load_memory_cases(args.memory_cases_file)
    turns: list[TurnMetric] = []
    details: list[dict[str, Any]] = []
    tp_total = 0
    fp_total = 0
    fn_total = 0
    stale_hits = 0
    recall_queries = 0

    for i, case in enumerate(cases):
        name = str(case.get("name", f"case-{i}"))
        setup_turns = [str(x) for x in case.get("setup_turns", []) if str(x).strip()]
        query = str(case.get("query", "")).strip()
        must_have = [str(x).lower() for x in case.get("must_have", []) if str(x).strip()]
        should_not_have = [str(x).lower() for x in case.get("should_not_have", []) if str(x).strip()]
        if not query:
            continue

        user = f"mem-user-{i}-{uuid.uuid4().hex[:8]}"
        session = f"mem-session-{i}-{uuid.uuid4().hex[:8]}"
        for prompt in setup_turns:
            metric, _ = await _run_turn(client, endpoint, prompt, session, user, args.timeout)
            turns.append(metric)

        metric, response = await _run_turn(client, endpoint, query, session, user, args.timeout)
        turns.append(metric)
        recall_queries += 1

        low = (response or "").lower()
        tp = sum(1 for term in must_have if term in low)
        fn = max(0, len(must_have) - tp)
        fp = sum(1 for term in should_not_have if term in low)
        stale = 1 if fp > 0 else 0

        tp_total += tp
        fn_total += fn
        fp_total += fp
        stale_hits += stale
        details.append(
            {
                "name": name,
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "stale_hit": bool(stale),
                "status_code": metric.status_code,
                "error": metric.error,
                "response_excerpt": response[:220].replace("\n", " "),
            }
        )

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    stale_rate = stale_hits / recall_queries if recall_queries else 0.0
    summary = {
        "endpoint": endpoint,
        "cases": len(details),
        "recall_queries": recall_queries,
        "tp_total": tp_total,
        "fp_total": fp_total,
        "fn_total": fn_total,
        "stale_hits": stale_hits,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "stale_error_rate": round(stale_rate, 4),
        "details": details,
    }
    return turns, summary


async def _run_groundedness_suite(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
) -> tuple[list[TurnMetric], dict[str, Any]]:
    endpoint = args.groundedness_endpoint
    cases = _load_groundedness_cases(args.groundedness_cases_file)
    turn_metrics: list[TurnMetric] = []
    gmetrics: list[GroundednessMetric] = []
    for i, case in enumerate(cases):
        prompt = str(case.get("prompt", "")).strip()
        if not prompt:
            continue
        session_id = f"grd-session-{i}-{uuid.uuid4().hex[:8]}"
        user_id = f"grd-user-{i % 3}"
        turn, response = await _run_turn(client, endpoint, prompt, session_id, user_id, args.timeout)
        turn_metrics.append(turn)
        if turn.error or turn.status_code != 200:
            gmetrics.append(
                GroundednessMetric(
                    endpoint=endpoint,
                    prompt=prompt,
                    score=0.0,
                    matched_expected=0,
                    expected_total=len(case.get("expected_terms", [])),
                    forbidden_hits=0,
                    has_sources=False,
                    status_code=turn.status_code,
                    error=turn.error or f"HTTP {turn.status_code}",
                    response_excerpt="",
                )
            )
            continue
        score, matched, expected_total, forbidden_hits, has_sources = _score_groundedness_response(case, response)
        excerpt = response[:240].replace("\n", " ")
        gmetrics.append(
            GroundednessMetric(
                endpoint=endpoint,
                prompt=prompt,
                score=round(score, 4),
                matched_expected=matched,
                expected_total=expected_total,
                forbidden_hits=forbidden_hits,
                has_sources=has_sources,
                status_code=turn.status_code,
                error=None,
                response_excerpt=excerpt,
            )
        )

    passed = sum(1 for m in gmetrics if m.score >= args.groundedness_pass_threshold and not m.error)
    scores = [m.score for m in gmetrics if not m.error]
    errors = sum(1 for m in gmetrics if m.error)
    summary = {
        "endpoint": endpoint,
        "cases": len(gmetrics),
        "pass_threshold": args.groundedness_pass_threshold,
        "passed": passed,
        "errors": errors,
        "pass_rate": round((passed / len(gmetrics)), 4) if gmetrics else 0.0,
        "avg_score": round((sum(scores) / len(scores)), 4) if scores else 0.0,
        "details": [asdict(m) for m in gmetrics],
    }
    return turn_metrics, summary


def _estimate_cost(
    turns: list[TurnMetric],
    input_cost_per_1m: float,
    output_cost_per_1m: float,
    provider_usage_delta: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    in_chars = sum(t.input_chars for t in turns)
    out_chars = sum(t.output_chars for t in turns)
    in_tokens = in_chars / 4.0
    out_tokens = out_chars / 4.0
    total_cost = ((in_tokens / 1_000_000.0) * input_cost_per_1m) + (
        (out_tokens / 1_000_000.0) * output_cost_per_1m
    )
    turns_count = max(1, len(turns))
    cost_per_1k_turns = (total_cost / turns_count) * 1000.0
    out = {
        "estimated_input_tokens": round(in_tokens, 2),
        "estimated_output_tokens": round(out_tokens, 2),
        "pricing_per_1m_tokens": {
            "input": input_cost_per_1m,
            "output": output_cost_per_1m,
        },
        "estimated_total_cost": round(total_cost, 6),
        "estimated_cost_per_1k_turns": round(cost_per_1k_turns, 6),
        "note": "Token estimate uses chars/4 heuristic and is an approximation.",
    }
    if provider_usage_delta:
        p_in = float(provider_usage_delta.get("provider_input_tokens", 0))
        p_out = float(provider_usage_delta.get("provider_output_tokens", 0))
        p_total = float(provider_usage_delta.get("provider_total_tokens", 0))
        p_cost = ((p_in / 1_000_000.0) * input_cost_per_1m) + ((p_out / 1_000_000.0) * output_cost_per_1m)
        p_per_1k = (p_cost / max(1, len(turns))) * 1000.0
        heuristic_total = in_tokens + out_tokens
        rel_err = None
        if p_total > 0:
            rel_err = round(abs(heuristic_total - p_total) / p_total, 4)
        out["provider_usage_delta"] = {
            "provider_input_tokens": int(p_in),
            "provider_output_tokens": int(p_out),
            "provider_total_tokens": int(p_total),
            "provider_total_cost": round(p_cost, 6),
            "provider_cost_per_1k_turns": round(p_per_1k, 6),
            "heuristic_vs_provider_token_relative_error": rel_err,
        }
    return out


def _extract_provider_usage_delta(
    health_before: Optional[dict[str, Any]],
    health_after: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if not health_before or not health_after:
        return None
    try:
        before = ((health_before.get("memory") or {}).get("llm_usage") or {})
        after = ((health_after.get("memory") or {}).get("llm_usage") or {})
        keys = ["provider_input_tokens", "provider_output_tokens", "provider_total_tokens"]
        if not all(k in before and k in after for k in keys):
            return None
        delta = {k: max(0, int(after.get(k, 0)) - int(before.get(k, 0))) for k in keys}
        if delta["provider_total_tokens"] == 0 and (delta["provider_input_tokens"] + delta["provider_output_tokens"]) == 0:
            return None
        return delta
    except Exception:
        return None


def _build_output_path(output_file: Optional[str]) -> str:
    if output_file:
        return output_file
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.makedirs("perf/results", exist_ok=True)
    return f"perf/results/eval_{ts}.json"


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    if args.dry_run:
        memory_cases = _load_memory_cases(args.memory_cases_file)
        cases = _load_groundedness_cases(args.groundedness_cases_file)
        return {
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_s": 0.0,
            "mode": args.mode,
            "memory_endpoint": args.memory_endpoint,
            "memory_cases": len(memory_cases),
            "groundedness_endpoint": args.groundedness_endpoint,
            "groundedness_cases": len(cases),
            "rounds": args.rounds,
            "dry_run": True,
            "message": "Dry run completed. No HTTP calls were made.",
        }

    async with httpx.AsyncClient(base_url=args.base_url) as client:
        health_before = await _fetch_health(client, args.timeout)
        latency_turns = await _run_latency_suite(client, args)
        memory_turns: list[TurnMetric] = []
        memory_summary = None
        if not args.skip_memory:
            memory_turns, memory_summary = await _run_memory_suite(client, args)
        groundedness_turns: list[TurnMetric] = []
        groundedness_summary = None
        if not args.skip_groundedness:
            groundedness_turns, groundedness_summary = await _run_groundedness_suite(client, args)
        health_after = await _fetch_health(client, args.timeout)

    all_turns = latency_turns + memory_turns + groundedness_turns
    by_endpoint: dict[str, list[TurnMetric]] = {}
    for turn in all_turns:
        by_endpoint.setdefault(turn.endpoint, []).append(turn)
    endpoint_summaries = {ep: _summarize_turns(turns) for ep, turns in by_endpoint.items()}

    finished_at = datetime.now(timezone.utc).isoformat()
    duration_s = round(time.perf_counter() - started, 3)
    output = {
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": duration_s,
        "base_url": args.base_url,
        "mode": args.mode,
        "memory_endpoint": args.memory_endpoint,
        "rounds": args.rounds,
        "dry_run": False,
        "summary": endpoint_summaries,
        "memory_quality": memory_summary,
        "groundedness": groundedness_summary,
        "cost_proxy": _estimate_cost(
            all_turns,
            args.input_cost_per_1m,
            args.output_cost_per_1m,
            provider_usage_delta=_extract_provider_usage_delta(health_before, health_after),
        ),
        "health_before": health_before,
        "health_after": health_after,
        "raw_turns": [asdict(t) for t in all_turns],
    }
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate latency, memory quality, and cost proxies.")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL.")
    parser.add_argument("--mode", choices=["chat", "stream", "both"], default="both")
    parser.add_argument("--rounds", type=int, default=10, help="Latency rounds per selected endpoint.")
    parser.add_argument("--timeout", type=float, default=65.0, help="Per-request timeout in seconds.")
    parser.add_argument(
        "--memory-endpoint",
        choices=["/chat", "/chat/stream"],
        default="/chat",
        help="Endpoint used for memory quality scenarios.",
    )
    parser.add_argument(
        "--memory-cases-file",
        default="perf/memory_cases.json",
        help="JSON file with memory quality cases.",
    )
    parser.add_argument("--skip-memory", action="store_true", help="Skip memory quality scenarios.")
    parser.add_argument(
        "--groundedness-endpoint",
        choices=["/chat", "/chat/stream"],
        default="/chat",
        help="Endpoint used for groundedness scenarios.",
    )
    parser.add_argument(
        "--groundedness-cases-file",
        default="perf/groundedness_cases.json",
        help="JSON file with groundedness cases.",
    )
    parser.add_argument(
        "--groundedness-pass-threshold",
        type=float,
        default=0.7,
        help="Score threshold for groundedness case pass.",
    )
    parser.add_argument("--skip-groundedness", action="store_true", help="Skip groundedness scenarios.")
    parser.add_argument("--input-cost-per-1m", type=float, default=0.0)
    parser.add_argument("--output-cost-per-1m", type=float, default=0.0)
    parser.add_argument("--output-file", default=None, help="Path to output JSON report.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without API calls.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = asyncio.run(_run(args))
    output_file = _build_output_path(args.output_file)
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved report: {output_file}")
    if output.get("dry_run"):
        print("Dry run: no endpoint requests were executed.")
    else:
        summary = output.get("summary", {})
        for endpoint, data in summary.items():
            p95 = data.get("latency_ms", {}).get("p95")
            ttft_p95 = data.get("first_token_ms", {}).get("p95")
            print(f"{endpoint} p95={p95} ms, ttft_p95={ttft_p95} ms, errors={data.get('errors')}")
        mem = output.get("memory_quality")
        if mem:
            print(
                "memory: recall={:.4f}, precision={:.4f}, stale_error_rate={:.4f}".format(
                    mem["recall"],
                    mem["precision"],
                    mem["stale_error_rate"],
                )
            )
        grd = output.get("groundedness")
        if grd:
            print(
                "groundedness: avg_score={:.4f}, pass_rate={:.4f}, errors={}".format(
                    grd["avg_score"],
                    grd["pass_rate"],
                    grd["errors"],
                )
            )
        cost = output.get("cost_proxy", {})
        print(f"estimated_cost_per_1k_turns={cost.get('estimated_cost_per_1k_turns')}")
        provider = cost.get("provider_usage_delta")
        if provider:
            print(
                "provider_cost_per_1k_turns={} (token_error={})".format(
                    provider.get("provider_cost_per_1k_turns"),
                    provider.get("heuristic_vs_provider_token_relative_error"),
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
