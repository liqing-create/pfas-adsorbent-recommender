"""Shared content-based matching for classification predictions and ground truth."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from typing import Any, Sequence


MIN_CONTENT_SIMILARITY = 0.82
MIN_FUZZY_TOKENS = 20
MIN_BEST_MATCH_MARGIN = 0.03
MIN_TABLE_CONTAINMENT = 0.95
MIN_TABLE_CONTAINMENT_TOKENS = 40
MIN_TABLE_CONTAINMENT_MARGIN = 0.10

_PUNCTUATION_TRANSLATION = str.maketrans(
    {
        "\u00a0": " ",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
)


@dataclass(frozen=True)
class ChunkMatch:
    source_index: int
    target_index: int
    similarity: float
    method: str


def normalize_content_tokens(value: Any) -> tuple[str, ...]:
    """Normalize harmless typography, case, and whitespace for comparison."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.translate(_PUNCTUATION_TRANSLATION).casefold()
    return tuple(re.findall(r"\w+|[^\w\s]", text))


def content_similarity(source: Any, target: Any) -> float:
    """Return token-sequence similarity in the inclusive range 0..1."""
    source_tokens = (
        source if isinstance(source, tuple) else normalize_content_tokens(source)
    )
    target_tokens = (
        target if isinstance(target, tuple) else normalize_content_tokens(target)
    )
    if not source_tokens or not target_tokens:
        return 0.0
    if source_tokens == target_tokens:
        return 1.0
    return SequenceMatcher(
        None,
        source_tokens,
        target_tokens,
        autojunk=False,
    ).ratio()


def normalize_table_match_tokens(value: Any) -> tuple[str, ...]:
    """Return formatting-insensitive words and values for table lineage."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.translate(_PUNCTUATION_TRANSLATION).casefold()
    raw_tokens = re.findall(r"[a-z]+|[-+]?\d+(?:\.\d+)?", text)
    tokens: list[str] = []
    for token in raw_tokens:
        if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", token):
            tokens.append(token)
            continue
        try:
            number = Decimal(token)
            normalized = format(number.normalize(), "f")
            tokens.append("0" if normalized in {"-0", "+0"} else normalized)
        except InvalidOperation:
            tokens.append(token)
    return tuple(tokens)


def table_containment_similarity(source: Any, target: Any) -> float:
    """Measure whether either table representation contains the other."""
    source_tokens = (
        source if isinstance(source, tuple) else normalize_table_match_tokens(source)
    )
    target_tokens = (
        target if isinstance(target, tuple) else normalize_table_match_tokens(target)
    )
    if not source_tokens or not target_tokens:
        return 0.0
    overlap = sum((Counter(source_tokens) & Counter(target_tokens)).values())
    return max(overlap / len(source_tokens), overlap / len(target_tokens))


def _file_type(record: dict) -> str:
    return str(record.get("file_type") or "").strip().casefold()


def _same_chunk_id(source: dict, target: dict) -> bool:
    return str(source.get("chunk_id")) == str(target.get("chunk_id"))


def duplicate_normalized_content_count(records: Sequence[dict]) -> int:
    counts: dict[tuple[str, tuple[str, ...]], int] = defaultdict(int)
    for record in records:
        tokens = normalize_content_tokens(record.get("enriched_text"))
        if tokens:
            counts[(_file_type(record), tokens)] += 1
    return sum(count - 1 for count in counts.values() if count > 1)


def match_chunks(
    source_records: Sequence[dict],
    target_records: Sequence[dict],
    *,
    min_similarity: float = MIN_CONTENT_SIMILARITY,
) -> list[ChunkMatch]:
    """
    Match source records to target records by content, one-to-one.

    Exact normalized-token matches are paired first. Remaining records may be
    paired fuzzily only within the same file type, above min_similarity,
    and when they are reciprocal, unambiguous best matches. Chunk IDs are used
    only to resolve otherwise equal candidate scores.
    """
    source_tokens = [
        normalize_content_tokens(record.get("enriched_text"))
        for record in source_records
    ]
    target_tokens = [
        normalize_content_tokens(record.get("enriched_text"))
        for record in target_records
    ]

    exact_targets: dict[tuple[str, tuple[str, ...]], deque[int]] = defaultdict(deque)
    for target_index, tokens in enumerate(target_tokens):
        if tokens:
            exact_targets[(_file_type(target_records[target_index]), tokens)].append(
                target_index
            )

    matches: list[ChunkMatch] = []
    unmatched_sources = set(range(len(source_records)))
    unmatched_targets = set(range(len(target_records)))

    for source_index, tokens in enumerate(source_tokens):
        if not tokens:
            continue
        candidates = exact_targets.get(
            (_file_type(source_records[source_index]), tokens)
        )
        if not candidates:
            continue
        while candidates and candidates[0] not in unmatched_targets:
            candidates.popleft()
        if not candidates:
            continue
        target_index = candidates.popleft()
        unmatched_sources.remove(source_index)
        unmatched_targets.remove(target_index)
        matches.append(ChunkMatch(source_index, target_index, 1.0, "exact"))

    # Precompute plausible fuzzy-pair scores once. The multiset-token Dice
    # score is an upper bound for SequenceMatcher's ordered-match ratio, so it
    # safely removes impossible candidates at much lower cost.
    source_counts = [Counter(tokens) for tokens in source_tokens]
    target_counts = [Counter(tokens) for tokens in target_tokens]
    pair_scores: dict[tuple[int, int], float] = {}
    prefilter_threshold = max(0.0, min_similarity - MIN_BEST_MATCH_MARGIN)
    for source_index in unmatched_sources:
        if len(source_tokens[source_index]) < MIN_FUZZY_TOKENS:
            continue
        for target_index in unmatched_targets:
            if _file_type(source_records[source_index]) != _file_type(
                target_records[target_index]
            ):
                continue
            if len(target_tokens[target_index]) < MIN_FUZZY_TOKENS:
                continue
            total_length = (
                len(source_tokens[source_index]) + len(target_tokens[target_index])
            )
            length_upper_bound = (
                2
                * min(
                    len(source_tokens[source_index]),
                    len(target_tokens[target_index]),
                )
                / total_length
            )
            if length_upper_bound < prefilter_threshold:
                continue
            token_overlap = sum(
                (source_counts[source_index] & target_counts[target_index]).values()
            )
            bag_upper_bound = 2 * token_overlap / total_length
            if bag_upper_bound < prefilter_threshold:
                continue
            pair_scores[(source_index, target_index)] = content_similarity(
                source_tokens[source_index],
                target_tokens[target_index],
            )

    while unmatched_sources and unmatched_targets:
        source_rankings: dict[int, list[tuple[float, bool, int]]] = {}
        target_rankings: dict[int, list[tuple[float, bool, int]]] = {}

        for source_index in unmatched_sources:
            source_candidates: list[tuple[float, bool, int]] = []
            for target_index in unmatched_targets:
                score = pair_scores.get((source_index, target_index))
                if score is None:
                    continue
                same_id = _same_chunk_id(
                    source_records[source_index],
                    target_records[target_index],
                )
                source_candidates.append((score, same_id, target_index))
                target_rankings.setdefault(target_index, []).append(
                    (score, same_id, source_index)
                )
            if source_candidates:
                source_rankings[source_index] = sorted(
                    source_candidates,
                    key=lambda item: (-item[0], -int(item[1]), item[2]),
                )

        for target_index, candidates in target_rankings.items():
            target_rankings[target_index] = sorted(
                candidates,
                key=lambda item: (-item[0], -int(item[1]), item[2]),
            )

        accepted: list[tuple[float, int, int]] = []
        for source_index, ranking in source_rankings.items():
            score, _same_id, target_index = ranking[0]
            if score < min_similarity:
                continue
            target_ranking = target_rankings[target_index]
            if target_ranking[0][2] != source_index:
                continue
            source_second = ranking[1][0] if len(ranking) > 1 else 0.0
            target_second = (
                target_ranking[1][0] if len(target_ranking) > 1 else 0.0
            )
            if score < 0.95 and (
                score - source_second < MIN_BEST_MATCH_MARGIN
                or score - target_second < MIN_BEST_MATCH_MARGIN
            ):
                continue
            accepted.append((score, source_index, target_index))

        if not accepted:
            break
        for score, source_index, target_index in sorted(accepted, reverse=True):
            if (
                source_index not in unmatched_sources
                or target_index not in unmatched_targets
            ):
                continue
            unmatched_sources.remove(source_index)
            unmatched_targets.remove(target_index)
            matches.append(
                ChunkMatch(source_index, target_index, score, "fuzzy")
            )

    # A repaired table may split one old GT record into multiple current
    # records. Match each still-unmatched current part independently to the old
    # table whose words and cell values contain it. Targets are intentionally
    # reusable here; source records remain one-to-one.
    table_target_tokens = {
        target_index: normalize_table_match_tokens(
            target_records[target_index].get("enriched_text")
        )
        for target_index in range(len(target_records))
        if _file_type(target_records[target_index]) == "table"
    }
    for source_index in sorted(tuple(unmatched_sources)):
        if _file_type(source_records[source_index]) != "table":
            continue
        tokens = normalize_table_match_tokens(
            source_records[source_index].get("enriched_text")
        )
        if len(tokens) < MIN_TABLE_CONTAINMENT_TOKENS:
            continue
        ranking = sorted(
            (
                (table_containment_similarity(tokens, target_tokens), target_index)
                for target_index, target_tokens in table_target_tokens.items()
                if len(target_tokens) >= MIN_TABLE_CONTAINMENT_TOKENS
            ),
            reverse=True,
        )
        if not ranking:
            continue
        best_score, target_index = ranking[0]
        second_score = ranking[1][0] if len(ranking) > 1 else 0.0
        if (
            best_score < MIN_TABLE_CONTAINMENT
            or best_score - second_score < MIN_TABLE_CONTAINMENT_MARGIN
        ):
            continue
        unmatched_sources.remove(source_index)
        matches.append(
            ChunkMatch(source_index, target_index, best_score, "table_containment")
        )

    return sorted(matches, key=lambda match: match.source_index)


__all__ = [
    "ChunkMatch",
    "MIN_CONTENT_SIMILARITY",
    "content_similarity",
    "duplicate_normalized_content_count",
    "match_chunks",
    "normalize_content_tokens",
    "normalize_table_match_tokens",
    "table_containment_similarity",
]
