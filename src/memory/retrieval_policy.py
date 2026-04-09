import re

_metrics = {
    "retrieval_attempts": 0,
    "retrieval_gate_skips": 0,
    "retrieval_timeouts": 0,
    "retrieval_failures": 0,
    "budget_applied": 0,
    "budget_item_drops": 0,
    "budget_char_truncations": 0,
}


def should_retrieve_long_term_memory(prompt: str) -> bool:
    """Decide if long-term memory retrieval is useful for this prompt."""
    if not prompt:
        return False
    text = prompt.strip().lower()
    if len(text) < 3:
        return False

    trivial = {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "ok",
        "okay",
        "cool",
        "nice",
    }
    if text in trivial:
        return False

    # Short follow-ups often rely on prior remembered context.
    short_followup_cues = [
        r"\bthat\b",
        r"\bthis\b",
        r"\bit\b",
        r"\bthose\b",
        r"\bwhat about\b",
        r"\band\b",
        r"\bagain\b",
        r"\bsame\b",
    ]
    if len(text) <= 24 and any(re.search(p, text) for p in short_followup_cues):
        return True

    # Strong indicators that user-specific memory is relevant.
    memory_cues = [
        r"\bmy\b",
        r"\bmine\b",
        r"\bremember\b",
        r"\bprefer\b",
        r"\bfavorite\b",
        r"\bi like\b",
        r"\bi love\b",
        r"\bwhat did i\b",
        r"\bwhat do i\b",
        r"\babout me\b",
    ]
    if any(re.search(p, text) for p in memory_cues):
        return True

    if len(text) < 8:
        return False

    # Retrieve for broader prompts where personalization may still help.
    return len(text.split()) >= 6


def adaptive_memory_limit(prompt: str, base: int = 4) -> int:
    """Set retrieval top-k based on query specificity."""
    text = (prompt or "").strip()
    words = len(text.split())
    if words <= 4:
        return 2
    if words <= 10:
        return base
    if words <= 20:
        return base + 1
    return min(base + 2, 8)


def apply_memory_budget(
    prompt: str,
    memories: list[str],
    *,
    max_items: int,
    max_chars: int,
    max_chars_per_item: int,
) -> list[str]:
    """Bound memory injection size to control prompt growth and latency."""
    _metrics["budget_applied"] += 1

    if not memories or max_items <= 0 or max_chars <= 0:
        return []

    text = (prompt or "").strip()
    words = len(text.split())
    dynamic_cap = max_items
    if words >= 28:
        dynamic_cap = max(1, max_items - 2)
    elif words >= 16:
        dynamic_cap = max(1, max_items - 1)

    unique: list[str] = []
    seen: set[str] = set()
    for raw in memories:
        item = " ".join(str(raw or "").split()).strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    kept: list[str] = []
    used_chars = 0
    for raw_item in unique:
        if len(kept) >= dynamic_cap:
            _metrics["budget_item_drops"] += 1
            continue

        item = raw_item
        if len(item) > max_chars_per_item:
            item = item[: max_chars_per_item - 3].rstrip() + "..."
            _metrics["budget_char_truncations"] += 1

        projected = used_chars + len(item) + (1 if kept else 0)
        if projected > max_chars:
            remaining = max_chars - used_chars - (1 if kept else 0)
            if remaining >= 24:
                item = item[: max(0, remaining - 3)].rstrip() + "..."
                kept.append(item)
                _metrics["budget_char_truncations"] += 1
                used_chars = max_chars
            else:
                _metrics["budget_item_drops"] += 1
            break

        kept.append(item)
        used_chars = projected

    return kept


def mark_retrieval_attempt() -> None:
    _metrics["retrieval_attempts"] += 1


def mark_retrieval_gate_skip() -> None:
    _metrics["retrieval_gate_skips"] += 1


def mark_retrieval_timeout() -> None:
    _metrics["retrieval_timeouts"] += 1


def mark_retrieval_failure() -> None:
    _metrics["retrieval_failures"] += 1


def get_retrieval_policy_metrics() -> dict:
    return dict(_metrics)
