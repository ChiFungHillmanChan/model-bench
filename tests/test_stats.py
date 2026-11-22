import pytest

from model_bench.stats import clustered_mean, mcnemar, paired_diff, pass_hat_k


def obs(item_scores):
    """{item: [epoch scores]} -> long format."""
    return [(i, e, v) for i, vs in item_scores.items() for e, v in enumerate(vs, start=1)]


def test_clustered_mean_known_values():
    est = clustered_mean(obs({"x": [1, 1], "y": [0, 0], "z": [1, 0]}))
    assert est.mean == pytest.approx(0.5)
    assert est.se == pytest.approx(0.5 / 3**0.5)
    assert est.ci_low == 0.0  # 0.5 - 1.96 * se is below 0, clipped
    assert est.ci_high == 1.0  # 0.5 + 1.96 * se is above 1, clipped
    assert (est.n_items, est.n_obs) == (3, 6)


def test_clustered_differs_from_naive_when_epochs_correlate():
    est = clustered_mean(obs({"x": [1] * 5, "y": [0] * 5}))
    assert est.n_items == 2 and est.n_obs == 10
    assert est.se == pytest.approx(0.5)  # sd([1,0]) / sqrt(2); naive over 10 obs would be ~0.17


def test_single_item_gives_none_ci():
    est = clustered_mean(obs({"x": [1, 0, 1]}))
    assert est.mean == pytest.approx(2 / 3)
    assert est.se is None and est.ci_low is None and est.ci_high is None


def test_empty_gives_none():
    est = clustered_mean([])
    assert est.mean is None and est.n_items == 0


def test_all_identical_scores_give_no_interval():
    est = clustered_mean(obs({"x": [1], "y": [1], "z": [1]}))
    assert est.mean == 1.0 and est.se == 0.0
    assert est.ci_low is None and est.ci_high is None  # zero-width CI would mislead (Review Focus 5)


def test_ci_clipped_to_unit_interval():
    est = clustered_mean(obs({str(i): [1] for i in range(9)} | {"z": [0]}))
    assert est.ci_high == 1.0 and est.ci_low >= 0.0


def test_paired_diff_identical_diffs_give_no_interval():
    est = paired_diff(obs({"1": [1], "2": [1]}), obs({"1": [0], "2": [0]}))
    assert est.mean == 1.0 and est.ci_low is None


def test_paired_diff_uses_shared_items_only():
    a = obs({"1": [1], "2": [1], "3": [0]})
    b = obs({"2": [0], "3": [0], "4": [1]})
    est = paired_diff(a, b)
    assert est.n_items == 2
    assert est.mean == pytest.approx(0.5)


def test_paired_diff_deterministic_seed():
    a = obs({str(i): [i % 2] for i in range(20)})
    b = obs({str(i): [(i // 2) % 2] for i in range(20)})
    assert paired_diff(a, b) == paired_diff(a, b)


def test_paired_diff_single_item_none_ci():
    est = paired_diff(obs({"1": [1]}), obs({"1": [0]}))
    assert est.mean == 1.0 and est.ci_low is None


def test_mcnemar_exact():
    a = {str(i): True for i in range(6)} | {"x": True, "y": False}
    b = {str(i): False for i in range(6)} | {"x": True, "y": False}
    b_only, a_only, p = mcnemar(a, b)
    assert (b_only, a_only) == (0, 6)
    assert p == pytest.approx(0.03125)


def test_mcnemar_no_discordant_is_none():
    assert mcnemar({"x": True}, {"x": True}) == (0, 0, None)


def test_pass_hat_k_none_when_epochs_1():
    assert pass_hat_k(obs({"x": [1], "y": [1]}), 1) is None
    assert pass_hat_k(obs({"x": [1], "y": [1]}), 3) is None


def test_pass_hat_k_value():
    data = obs({"x": [1, 1, 1], "y": [1, 0, 1], "z": [1, 1, 1], "w": [0, 0, 0]})
    assert pass_hat_k(data, 3) == pytest.approx(0.5)
    assert pass_hat_k(data, 2) == pytest.approx(0.5)
