"""Validation, ledger mapping and merging of the analyst's structured output.

Models return near-valid structures: indices out of range, a status contradicted by its
own citation list, a strength no source can support. Deciding which side of each
contradiction wins is policy, not prompting, so it lives here as pure functions and
never inside the node that calls the model.
"""

import logging
from collections.abc import Sequence
from typing import NamedTuple

from agents.research_team.contracts import (
    AspectCoverage,
    EvidenceCoverage,
    EvidenceStatement,
    ResearchSubject,
    ResearchTask,
    SubjectEvidenceCoverage,
)
from core.metrics import Degradation, record_degradation

logger = logging.getLogger(__name__)

# Merged output is re-sent on every call, so it needs a ceiling. Without one the
# prompt grows with the number of retrieval rounds instead of with the evidence.
MAX_MERGED_FACTS = 60
MAX_MERGED_PATTERNS = 8
MAX_MERGED_CONTRADICTIONS = 10
MAX_MERGED_GAPS = 12
MAX_MERGED_STATEMENTS = 80


def _valid_indices(
    indices: Sequence[int], source_count: int, *, unique: bool
) -> tuple[list[int], bool]:
    """Keep in-range citations; report whether anything had to be dropped.

    ``unique`` follows the caller's contract: task-level coverage keeps the model's
    order and duplicates, the per-aspect and statement lists are sets.
    """
    kept = [index for index in indices if 1 <= index <= source_count]
    if not unique:
        return kept, len(kept) != len(indices)
    deduped = sorted(set(kept))
    return deduped, len(deduped) != len(set(indices))


class _StrengthKeys(NamedTuple):
    """The three repairs this rule can make, named per coverage level."""

    without_sources: Degradation
    conflicting: Degradation


_COVERAGE_KEYS = _StrengthKeys(
    Degradation.COVERAGE_STRENGTH_WITHOUT_SOURCES,
    Degradation.COVERAGE_CONFLICTING_STRENGTH_MISMATCH,
)
_ASPECT_KEYS = _StrengthKeys(
    Degradation.SUBJECT_COVERAGE_STRENGTH_WITHOUT_SOURCES,
    Degradation.SUBJECT_COVERAGE_CONFLICTING_STRENGTH_MISMATCH,
)


def _reconcile_strength(
    status: str, strength: str, has_sources: bool, keys: _StrengthKeys, label: str
) -> str:
    """Force strength into what the status allows, leaving a trace each time.

    Callers clear the citations of a `missing` row before getting here, so "missing"
    always arrives without sources and the first rule already covers it. That is why
    there is no separate missing-strength rule: it used to exist and could never fire.
    """
    if not has_sources and strength != "insufficient":
        logger.warning("%s has no sources but strength %s", label, strength)
        record_degradation(keys.without_sources)
        return "insufficient"
    if status == "conflicting" and strength == "strong":
        logger.warning("%s is conflicting but strength is strong", label)
        record_degradation(keys.conflicting)
        return "moderate"
    return strength


def normalize_coverage(
    coverage: list[EvidenceCoverage],
    tasks: list[ResearchTask],
    source_count: int,
) -> list[EvidenceCoverage]:
    by_task: dict[str, EvidenceCoverage] = {}
    for item in coverage:
        if item.task_id not in {task.id for task in tasks}:
            logger.warning("Analyst returned coverage for unknown task %s", item.task_id)
            record_degradation(Degradation.COVERAGE_UNKNOWN_TASK)
            continue
        if item.task_id in by_task:
            logger.warning("Analyst returned duplicate coverage for task %s", item.task_id)
            record_degradation(Degradation.COVERAGE_DUPLICATE_TASK)
            continue
        valid_indices, invalid = _valid_indices(item.source_indices, source_count, unique=False)
        if invalid:
            logger.warning("Analyst returned invalid source indices for task %s", item.task_id)
            record_degradation(Degradation.COVERAGE_INVALID_SOURCE_INDEX)
        if item.status == "missing" and valid_indices:
            # "missing" means no supporting evidence, so a citation list contradicts it.
            # The status wins: rendering "缺失，支持来源 [1][4][6]" to the writer is worse
            # than losing the anchors, which evidence_statements still carry.
            logger.warning("Coverage task %s is missing but cites sources", item.task_id)
            record_degradation(Degradation.COVERAGE_MISSING_WITH_SOURCES)
            valid_indices = []
        if source_count > 1 and len(valid_indices) == source_count:
            # "Covered by every page we were shown" is not a comparison. Coverage gets
            # coarser as the window widens, so the rate of this is the signal to watch -
            # not an error, but a hint that each task needs a smaller candidate set.
            record_degradation(Degradation.COVERAGE_ALL_SOURCES_CITED)
        strength = _reconcile_strength(
            item.status,
            item.evidence_strength,
            bool(valid_indices),
            _COVERAGE_KEYS,
            f"Coverage task {item.task_id}",
        )
        by_task[item.task_id] = item.model_copy(
            update={"source_indices": valid_indices, "evidence_strength": strength}
        )

    normalized: list[EvidenceCoverage] = []
    for task in tasks:
        item = by_task.get(task.id)
        if item is None:
            logger.warning("Analyst omitted coverage for task %s", task.id)
            record_degradation(Degradation.COVERAGE_INCOMPLETE)
            normalized.append(
                EvidenceCoverage(
                    task_id=task.id,
                    question=task.question,
                    source_indices=[],
                    status="missing",
                    evidence_strength="insufficient",
                    gaps=["未生成该子问题的覆盖判断"],
                    reason="Analyst 未返回该子问题的覆盖记录",
                )
            )
            continue
        if item.question != task.question:
            item = item.model_copy(update={"question": task.question})
        normalized.append(item)
    return normalized


_STRENGTH_ORDER = {"insufficient": 0, "weak": 1, "moderate": 2, "strong": 3}


def _normalize_aspects(item: SubjectEvidenceCoverage, source_count: int) -> list[AspectCoverage]:
    """Apply the per-aspect rules: drop bad names, repair indices and strength."""
    aspects: list[AspectCoverage] = []
    seen: set[str] = set()
    for aspect in item.aspects:
        name = aspect.name.strip()
        if not name:
            record_degradation(Degradation.SUBJECT_COVERAGE_EMPTY_ASPECT)
            continue
        if name in seen:
            record_degradation(Degradation.SUBJECT_COVERAGE_DUPLICATE_ASPECT)
            continue
        seen.add(name)
        valid_indices, invalid = _valid_indices(aspect.source_indices, source_count, unique=True)
        if invalid:
            record_degradation(Degradation.SUBJECT_COVERAGE_INVALID_SOURCE_INDEX)
        if aspect.status == "missing" and valid_indices:
            record_degradation(Degradation.SUBJECT_COVERAGE_MISSING_WITH_SOURCES)
            valid_indices = []
        strength = _reconcile_strength(
            aspect.status,
            aspect.evidence_strength,
            bool(valid_indices),
            _ASPECT_KEYS,
            f"Aspect {item.subject_id}/{name}",
        )
        aspects.append(
            aspect.model_copy(
                update={
                    "name": name,
                    "source_indices": valid_indices,
                    "evidence_strength": strength,
                }
            )
        )
    return aspects


def _aggregate_subject(aspects: Sequence[AspectCoverage]) -> tuple[str, str]:
    """Fold aspects into one verdict: worst status, weakest supporting strength.

    `conflicting` never reaches the strength cap here: any conflicting aspect was
    already capped at `moderate` by its own rule, and the fold takes the minimum.
    """
    statuses = [aspect.status for aspect in aspects]
    if not statuses or all(status == "missing" for status in statuses):
        return "missing", "insufficient"
    if "conflicting" in statuses:
        status = "conflicting"
    elif any(status in {"missing", "partial"} for status in statuses):
        status = "partial"
    else:
        status = "sufficient"
    strength = min(
        (aspect.evidence_strength for aspect in aspects),
        key=lambda value: _STRENGTH_ORDER[value],
    )
    return status, strength


def normalize_subject_coverage(
    coverage: list[SubjectEvidenceCoverage],
    subjects: list[ResearchSubject],
    source_count: int,
) -> list[SubjectEvidenceCoverage]:
    known = {subject.id: subject for subject in subjects}
    by_subject: dict[str, SubjectEvidenceCoverage] = {}
    for item in coverage:
        subject = known.get(item.subject_id)
        if subject is None:
            logger.warning("Analyst returned coverage for unknown subject %s", item.subject_id)
            record_degradation(Degradation.SUBJECT_COVERAGE_UNKNOWN_SUBJECT)
            continue
        if item.subject_id in by_subject:
            logger.warning("Analyst returned duplicate coverage for subject %s", item.subject_id)
            record_degradation(Degradation.SUBJECT_COVERAGE_DUPLICATE_SUBJECT)
            continue

        aspects = _normalize_aspects(item, source_count)
        own_indices, _ = _valid_indices(item.source_indices, source_count, unique=True)
        subject_indices = sorted(
            set(own_indices) | {index for aspect in aspects for index in aspect.source_indices}
        )
        if len(subject_indices) != len(own_indices):
            record_degradation(Degradation.SUBJECT_COVERAGE_ASPECT_SOURCE_MERGE)

        status, strength = _aggregate_subject(aspects)
        if status == "missing":
            # Same invariant as task-level coverage: a missing object cannot have
            # supporting sources, even when the model listed some at object level.
            subject_indices = []

        gaps = list(dict.fromkeys(item.gaps + [gap for aspect in aspects for gap in aspect.gaps]))
        by_subject[item.subject_id] = item.model_copy(
            update={
                "subject": subject.name,
                "aspects": aspects,
                "source_indices": subject_indices,
                "status": status,
                "evidence_strength": strength,
                "gaps": gaps,
            }
        )

    normalized: list[SubjectEvidenceCoverage] = []
    for subject in subjects:
        item = by_subject.get(subject.id)
        if item is None:
            logger.warning("Analyst omitted coverage for subject %s", subject.id)
            record_degradation(Degradation.SUBJECT_COVERAGE_INCOMPLETE)
            normalized.append(
                SubjectEvidenceCoverage(
                    subject_id=subject.id,
                    subject=subject.name,
                    aspects=[],
                    status="missing",
                    evidence_strength="insufficient",
                    source_indices=[],
                    gaps=["未生成该对象的覆盖判断"],
                    reason="Analyst 未返回该对象的覆盖记录",
                )
            )
        else:
            normalized.append(item)
    return normalized


def normalize_evidence_statements(
    statements: list[EvidenceStatement],
    source_count: int,
) -> list[EvidenceStatement]:
    normalized: list[EvidenceStatement] = []
    for item in statements:
        statement = item.statement.strip()
        if not statement:
            record_degradation(Degradation.EVIDENCE_STATEMENT_EMPTY)
            continue
        source_indices, invalid = _valid_indices(item.source_indices, source_count, unique=True)
        if not source_indices:
            record_degradation(Degradation.EVIDENCE_STATEMENT_WITHOUT_SOURCES)
            continue
        if invalid:
            record_degradation(Degradation.EVIDENCE_STATEMENT_INVALID_SOURCE_INDEX)

        periods = list(dict.fromkeys(period.strip() for period in item.periods if period.strip()))
        if item.period and item.period.strip() not in periods:
            periods.insert(0, item.period.strip())
        kind = item.kind
        trend_origin = item.trend_origin
        if kind == "trend":
            if item.value_type in {"planned", "forecast"}:
                kind = "forecast"
                trend_origin = "unknown"
                record_degradation(Degradation.EVIDENCE_STATEMENT_FORECAST_NOT_TREND)
                normalized.append(
                    item.model_copy(
                        update={
                            "statement": statement,
                            "kind": kind,
                            "period": periods[0] if periods else item.period,
                            "periods": periods,
                            "metric": item.metric.strip() if item.metric else item.metric,
                            "unit": item.unit.strip() if item.unit else item.unit,
                            "source_indices": source_indices,
                            "trend_origin": trend_origin,
                        }
                    )
                )
                continue
            explicit = trend_origin == "explicit_source_statement"
            comparable = (
                len(periods) >= 3
                and item.value_type in {"actual", "reported"}
                and bool(item.metric and item.metric.strip())
                and bool(item.unit and item.unit.strip())
            )
            if not explicit and not comparable:
                kind = "change" if len(periods) >= 2 else "point"
                trend_origin = "unknown"
                record_degradation(Degradation.EVIDENCE_STATEMENT_TREND_DOWNGRADED)
        elif kind == "change" and len(periods) < 2:
            kind = "point"
            record_degradation(Degradation.EVIDENCE_STATEMENT_CHANGE_DOWNGRADED)
        elif kind == "forecast" and item.value_type not in {"planned", "forecast"}:
            kind = "change" if len(periods) >= 2 else "point"
            record_degradation(Degradation.EVIDENCE_STATEMENT_FORECAST_DOWNGRADED)

        normalized.append(
            item.model_copy(
                update={
                    "statement": statement,
                    "kind": kind,
                    "period": periods[0] if periods else item.period,
                    "periods": periods,
                    "metric": item.metric.strip() if item.metric else item.metric,
                    "unit": item.unit.strip() if item.unit else item.unit,
                    "source_indices": source_indices,
                    "trend_origin": trend_origin,
                }
            )
        )
    return normalized


def to_ledger_indices(indices: Sequence[int], mapping: dict[int, int]) -> list[int]:
    """Batch numbering -> ledger numbering, dropping numbers outside the batch."""
    return sorted({mapping[index] for index in indices if index in mapping})


def to_ledger_coverage(
    coverage: Sequence[EvidenceCoverage], mapping: dict[int, int]
) -> list[EvidenceCoverage]:
    return [
        item.model_copy(update={"source_indices": to_ledger_indices(item.source_indices, mapping)})
        for item in coverage
    ]


def to_ledger_subject_coverage(
    coverage: Sequence[SubjectEvidenceCoverage], mapping: dict[int, int]
) -> list[SubjectEvidenceCoverage]:
    return [
        item.model_copy(
            update={
                "source_indices": to_ledger_indices(item.source_indices, mapping),
                "aspects": [
                    aspect.model_copy(
                        update={"source_indices": to_ledger_indices(aspect.source_indices, mapping)}
                    )
                    for aspect in item.aspects
                ],
            }
        )
        for item in coverage
    ]


def to_ledger_statements(
    statements: Sequence[EvidenceStatement], mapping: dict[int, int]
) -> list[EvidenceStatement]:
    return [
        item.model_copy(update={"source_indices": to_ledger_indices(item.source_indices, mapping)})
        for item in statements
    ]


def merge_lines(previous: Sequence[str], current: Sequence[str], limit: int) -> list[str]:
    """Union two lists oldest-first, deduped on whitespace-normalised text.

    Anchors are already in ledger numbering when this runs, so the same fact seen in
    two batches collapses instead of being stored twice under [1] and [13].
    """
    merged = [item.strip() for item in previous if item.strip()]
    seen = {" ".join(item.split()) for item in merged}
    for item in current:
        text = item.strip()
        key = " ".join(text.split())
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(text)
    return merged[-limit:]


def merge_coverage(
    previous: Sequence[EvidenceCoverage], current: Sequence[EvidenceCoverage]
) -> list[EvidenceCoverage]:
    """Keep an earlier judgement unless the new batch actually cites something.

    A call only sees its own batch, so a task whose evidence arrived earlier looks
    uncovered. Reading that as a downgrade is exactly how a wider window would turn
    into a regression: the model would "forget" what it is no longer shown.
    """
    by_task = {item.task_id: item for item in previous}
    merged: list[EvidenceCoverage] = []
    for item in current:
        old = by_task.get(item.task_id)
        if old is not None and not item.source_indices and old.source_indices:
            record_degradation(Degradation.COVERAGE_STALE_BATCH)
            merged.append(old.model_copy(update={"question": item.question}))
            continue
        merged.append(item)
    # No tail merge: normalize_coverage emits a row for every current task, so a
    # leftover here would be a task id that no longer exists. Carrying it forward would
    # put a stale judgement in front of the writer.
    return merged


def merge_subject_coverage(
    previous: Sequence[SubjectEvidenceCoverage], current: Sequence[SubjectEvidenceCoverage]
) -> list[SubjectEvidenceCoverage]:
    """Same ratchet as :func:`merge_coverage`, at object granularity."""
    by_subject = {item.subject_id: item for item in previous}
    merged: list[SubjectEvidenceCoverage] = []
    for item in current:
        old = by_subject.get(item.subject_id)
        if old is not None and not item.source_indices and old.source_indices:
            record_degradation(Degradation.SUBJECT_COVERAGE_STALE_BATCH)
            merged.append(old.model_copy(update={"subject": item.subject}))
            continue
        merged.append(item)
    # Same reasoning as merge_coverage: every current subject is normalised in.
    return merged


def _statement_key(item: EvidenceStatement) -> tuple[str, str, str, str]:
    return (
        item.kind,
        item.metric or "",
        "、".join(item.periods) or item.period or "",
        " ".join(item.statement.split()),
    )


def merge_statements(
    previous: Sequence[EvidenceStatement], current: Sequence[EvidenceStatement]
) -> list[EvidenceStatement]:
    seen = {_statement_key(item) for item in previous}
    merged = list(previous)
    for item in current:
        key = _statement_key(item)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged[-MAX_MERGED_STATEMENTS:]
