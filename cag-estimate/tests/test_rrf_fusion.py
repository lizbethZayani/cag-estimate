"""Tests for reciprocal rank fusion (pure, no I/O)."""

import pytest

from cag_estimate.embedding_pipeline.fusion import reciprocal_rank_fusion


def test_single_ranking_keeps_its_order() -> None:
    fused = reciprocal_rank_fusion([[7, 3, 9]])

    assert [chunk_id for chunk_id, _ in fused] == [7, 3, 9]


def test_item_in_both_rankings_beats_items_in_one() -> None:
    fused = reciprocal_rank_fusion([[1, 2, 3], [4, 5, 3]])

    assert fused[0][0] == 3


def test_scores_match_the_hand_computed_formula() -> None:
    fused = dict(reciprocal_rank_fusion([[1, 2], [2, 3]], k=60))

    assert fused[1] == pytest.approx(1 / 61)
    assert fused[2] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[3] == pytest.approx(1 / 62)


def test_output_is_sorted_by_score_descending() -> None:
    fused = reciprocal_rank_fusion([[1, 2], [2, 3]])

    scores = [score for _, score in fused]
    assert scores == sorted(scores, reverse=True)
    assert [chunk_id for chunk_id, _ in fused] == [2, 1, 3]


def test_score_ties_prefer_the_best_rank_achieved() -> None:
    # With k=0 chunk 1 scores 1/2 + 1/2 = 1.0 and chunks 7 and 8 score 1/1 = 1.0:
    # all tie, so the two rank-1 chunks go before chunk 1 although 1 appears first.
    fused = reciprocal_rank_fusion([[7, 1], [8, 1]], k=0)

    assert [chunk_id for chunk_id, _ in fused] == [7, 8, 1]


def test_remaining_ties_follow_first_appearance() -> None:
    fused = reciprocal_rank_fusion([[1, 2], [4, 3]], k=60)

    assert [chunk_id for chunk_id, _ in fused] == [1, 4, 2, 3]


def test_equal_score_and_best_rank_fall_back_to_first_appearance() -> None:
    fused = reciprocal_rank_fusion([[5], [6]])

    assert [chunk_id for chunk_id, _ in fused] == [5, 6]


def test_result_is_deterministic() -> None:
    rankings = [[3, 1, 2], [2, 4, 1], [5, 3]]

    assert reciprocal_rank_fusion(rankings) == reciprocal_rank_fusion(rankings)


@pytest.mark.parametrize("rankings", [[], [[]], [[], []]])
def test_empty_input_gives_empty_output(rankings: list[list[int]]) -> None:
    assert reciprocal_rank_fusion(rankings) == []


def test_k_changes_the_scores_but_not_necessarily_the_order() -> None:
    rankings = [[1, 2, 3], [1, 3, 2]]

    small = reciprocal_rank_fusion(rankings, k=1)
    large = reciprocal_rank_fusion(rankings, k=100)

    assert [c for c, _ in small] == [c for c, _ in large]
    assert small[0][1] != pytest.approx(large[0][1])


def test_k_zero_is_legal() -> None:
    fused = dict(reciprocal_rank_fusion([[1, 2]], k=0))

    assert fused[1] == pytest.approx(1.0)
    assert fused[2] == pytest.approx(0.5)


def test_negative_k_is_rejected() -> None:
    with pytest.raises(ValueError, match="k"):
        reciprocal_rank_fusion([[1]], k=-1)
