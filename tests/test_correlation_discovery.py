from unittest.mock import Mock

import numpy as np
import pytest
from scipy.stats import kendalltau, pearsonr, spearmanr

from tau_eval import visualization


@pytest.fixture
def observations():
    return {
        "first": {
            "z": {"x": 0.8, "y": 0.1},
            "a": {"x": 0.1, "y": 0.7},
            "b": {"x": 0.4, "y": 0.4},
            "missing": {"x": 0.3},
            "bad": {"x": "not numeric", "y": 0.2},
            "nan": {"x": 0.5, "y": np.nan},
            "original_metrics": {"x": 0.0, "y": 1.0},
        },
        "second": {
            "a": {"x": 0.2, "y": 0.3},
            "b": {"x": 0.5, "y": 0.9},
            "z": {"x": 0.7, "y": 0.6},
            "bad": "not a metric dictionary",
        },
        "empty": {},
        "constant": {"a": {"x": 1.0, "y": 0.2}, "b": {"x": 1.0, "y": 0.5}},
        "single": {"a": {"x": 0.1, "y": 0.9}},
    }


@pytest.mark.parametrize("aggregate", [False, True])
@pytest.mark.parametrize(
    "method, reference", [("kendall", kendalltau), ("pearson", pearsonr), ("spearman", spearmanr)]
)
def test_model_discovery_once_preserves_statistics(observations, monkeypatch, aggregate, method, reference):
    discover = Mock(wraps=visualization.get_all_model_method_names)
    monkeypatch.setattr(visualization, "get_all_model_method_names", discover)
    data = {key: observations[key] for key in ["first", "second", "empty"]}

    result = visualization.compute_correlation(data, "x", "y", method=method, aggregate_across_tasks=aggregate)

    discover.assert_called_once_with(data)
    first_x, first_y = [0.1, 0.4, 0.8], [0.7, 0.4, 0.1]
    second_x, second_y = [0.2, 0.5, 0.7], [0.3, 0.9, 0.6]
    if aggregate:
        np.testing.assert_allclose(result["aggregated_tasks"], reference(first_x + second_x, first_y + second_y))
    else:
        np.testing.assert_allclose(result["first"], reference(first_x, first_y))
        np.testing.assert_allclose(result["second"], reference(second_x, second_y))
        assert all(np.isnan(value) for value in result["empty"])


@pytest.mark.parametrize("task", ["empty", "constant", "single", "missing"])
def test_missing_and_insufficient_task_data_remain_nan(observations, task):
    result = visualization.compute_correlation(observations, "x", "y", task_name=task)

    assert list(result) == [task]
    assert all(np.isnan(value) for value in result[task])


def test_discovery_is_not_cached_across_requests(monkeypatch):
    data = {"task": {"a": {"x": 0.2, "y": 0.8}}}
    discover = Mock(wraps=visualization.get_all_model_method_names)
    monkeypatch.setattr(visualization, "get_all_model_method_names", discover)

    first = visualization.compute_correlation(data, "x", "y")
    data["task"]["new_model"] = {"x": 0.6, "y": 0.1}
    second = visualization.compute_correlation(data, "x", "y")

    assert discover.call_count == 2
    assert all(np.isnan(value) for value in first["task"])
    np.testing.assert_allclose(second["task"], kendalltau([0.2, 0.6], [0.8, 0.1]))
