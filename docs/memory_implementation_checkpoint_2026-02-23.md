# Memory Implementation Checkpoint

Date: 2026-02-23
Project: `langgraph_chatbot`

## Objective
Implement efficient, production-grade chatbot memory beyond sliding window context using a layered memory architecture, retrieval controls, and cost-aware operations.

## Target Architecture
1. `L0` Short-term context: keep sliding window in prompt.
2. `L1` Session memory: DB-backed session/checkpoint state (resumable per session/thread).
3. `L2` Long-term user memory: semantic memory store (vector-capable, user-scoped).
4. `L3` Knowledge RAG: separate domain/corpus retrieval from user memory.

## Storage Strategy
1. Start with Postgres + pgvector for simple ops + joins with relational user/session data.
2. Upgrade to Qdrant/Weaviate if/when advanced retrieval/scale features are needed.
3. Use Redis for hot cache and latency optimization, not sole long-term memory.

## Retrieval Strategy
1. Hybrid retrieval: dense + sparse/BM25.
2. Rerank top candidates before prompt injection.
3. User/session namespace filters + recency weighting.
4. Retrieval gate (retrieve-or-not) to avoid unnecessary context expansion.
5. Hierarchical/graph retrieval patterns for broad corpus questions.

## Memory Write Strategy
1. Do not store all turns as long-term memory.
2. Store only extracted durable memory units:
   - preferences
   - profile facts
   - ongoing tasks/commitments
   - stable decisions
3. Keep provenance metadata:
   - source message
   - timestamp
   - confidence
4. Apply dedupe, TTL/decay, and periodic compaction summaries.

## Session-wise Data Model
1. `sessions`
2. `messages` (raw auditable history)
3. `session_summaries` (rolling compression)
4. `user_memories` (semantic memory units + embedding refs)
5. `memory_events` (upsert/merge/delete audit trail)

## Cost & Efficiency Principles
1. Token usage typically dominates cost more than vector storage.
2. Control retrieval token bloat with adaptive top-k, reranking, and strict budgets.
3. Use compaction/summarization and memory write filtering to reduce storage and prompt size growth.

## Implementation Phases
### Phase 1 (3-5 days)
- DB-backed session persistence.
- Rolling session summaries.
- Basic memory telemetry.

### Phase 2 (4-7 days)
- Long-term memory extraction pipeline.
- Vector store integration + user/session filters.

### Phase 3 (4-6 days)
- Hybrid retrieval.
- Reranker.
- Retrieval gate.

### Phase 4 (2-4 days)
- Cost controls: token budgets, adaptive k, cache policies.
- Performance tuning for latency/cost tradeoff.

### Phase 5 (3-5 days)
- Evaluation harness:
  - groundedness
  - memory precision/recall
  - stale memory error rate
  - p95 latency
  - cost per 1k turns
- Rollout hardening and monitoring.

## Delivery Estimate
1. MVP (usable): 2-3 weeks.
2. Production-grade (with eval + cost controls): 3-5 weeks.
3. Fast path v1 (Postgres + pgvector first): ~10-14 days.

## Execution Notes
1. Keep user memory and domain RAG separated.
2. Keep memory writes selective and auditable.
3. Optimize for recall quality and cost simultaneously (not one without the other).

## Progress
- [x] Phase 1 started
- [x] Session persistence service scaffolded with MongoDB backend + in-process fallback
- [x] Rolling summary persistence (interval-based) added
- [x] Memory telemetry added to health output
- [ ] Phase 1 validation under real deployment traffic
- [x] Phase 2 started
- [x] Long-term memory store scaffolded with scoped persistence + retrieval
- [x] Long-term memory retrieval wired into graph input
- [x] Long-term memory extraction/write wired after each turn
- [x] Phase 2 vector DB backend upgrade scaffolded (pgvector + fallback strategy)
- [x] Phase 3 started
- [x] Retrieval gate + adaptive top-k policy wired into request flow
- [x] Hybrid-style ranking signal added (semantic + lexical + recency + confidence)
- [x] Reranking pass added for memory candidates (query-aware boosts)
- [x] Conflict resolution added for slot-like memories (new fact supersedes prior active fact)
- [x] Phase 4 started
- [x] Memory injection budget controls added (max items + char budgets + per-item truncation)
- [x] Timeout-protected long-term memory retrieval to avoid tail-latency spikes
- [x] Timeout-protected session/memory persistence with background writes on stream endpoint
- [x] DB circuit-breaker fallback added for memory/session stores on repeated timeout/failure paths
- [x] Cache policy hardened (scope-aware keys + volatile intent cache bypass)
- [x] Phase 4 telemetry added (retrieval policy + runtime timeout/failure counters in health)
- [x] Phase 5 started
- [x] Evaluation harness added (`perf/eval_harness.py`) for p95/p99 latency + memory quality + cost proxy
- [x] Groundedness scoring integrated into automated harness (case-based scoring in `perf/groundedness_cases.json`)
- [x] Memory precision/recall + stale error scoring integrated into harness (dataset-driven via `perf/memory_cases.json`)
- [x] Session summary context now injected into generation for long-conversation continuity
- [x] Multi-fact memory extraction per turn added (name/location/preferences/tasks in one message)
- [x] Retrieval fallback improved for short follow-ups and low-similarity misses
- [ ] Memory precision/recall and stale error calibration on production-like datasets
- [ ] Cost per 1k turns validated against provider-side token usage telemetry
- [x] Provider token-usage telemetry aggregation added to health (`memory.llm_usage`) and harness delta comparison support
