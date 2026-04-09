# Load Testing

## k6 stream latency test

Run:

```bash
k6 run perf/k6-stream-latency.js
```

With remote host:

```bash
BASE_URL=https://your-host k6 run perf/k6-stream-latency.js
```

## Locust mixed traffic test

Install dev dependency:

```bash
pip install -r requirements-dev.txt
```

Run:

```bash
locust -f perf/locustfile.py --host http://localhost:8000
```

## Tuning workflow

1. Run k6 and Locust against staging with representative prompts.
2. Watch p95/p99 latency and error rate.
3. Adjust `--workers`, `--limit-concurrency`, and API timeouts.
4. Re-run until p95 and p99 stabilize under your SLO.

## Phase 5 evaluation harness

Run a combined latency + memory-quality + cost-proxy report:

```bash
python perf/eval_harness.py --base-url http://localhost:8000 --mode both --rounds 12
```

Use `/chat/stream` for memory scenarios:

```bash
python perf/eval_harness.py --memory-endpoint /chat/stream
```

Use custom memory evaluation cases:

```bash
python perf/eval_harness.py \
  --memory-endpoint /chat \
  --memory-cases-file perf/memory_cases.json
```

Run groundedness scenarios with custom case file:

```bash
python perf/eval_harness.py \
  --groundedness-endpoint /chat \
  --groundedness-cases-file perf/groundedness_cases.json \
  --groundedness-pass-threshold 0.7
```

Include pricing inputs (for estimated cost per 1k turns):

```bash
python perf/eval_harness.py \
  --input-cost-per-1m 0.15 \
  --output-cost-per-1m 0.60
```

If `/health` exposes `memory.llm_usage`, the harness also computes provider-token
cost-per-1k-turns and compares heuristic token estimates vs provider token deltas.

Dry run (no network calls):

```bash
python perf/eval_harness.py --dry-run
```

Reports are written to `perf/results/eval_<timestamp>.json`.
