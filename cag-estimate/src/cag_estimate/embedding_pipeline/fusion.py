"""Reciprocal rank fusion: merge several best-first rankings into one."""

from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]], *, k: int = 60
) -> list[tuple[int, float]]:
    """Fuse rankings of chunk ids (best first) into ``(chunk_id, score)`` pairs, best first.

    The score of an id is the sum over the rankings that contain it of ``1 / (k + rank)``
    with a 1-based rank (the original RRF definition). ``k`` must be >= 0 (0 is legal and
    makes the top ranks dominate). Ties on the score are broken by the best (lowest) rank
    the id achieved in any ranking, then by first appearance, so the output is deterministic.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    scores: dict[int, float] = {}
    best_rank: dict[int, int] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            best_rank[chunk_id] = min(best_rank.get(chunk_id, rank), rank)
    # dict preserves first appearance and sorted is stable, so it is the last tie-break.
    return sorted(scores.items(), key=lambda item: (-item[1], best_rank[item[0]]))
