import matplotlib.pyplot as plt
import numpy as np
import pytest

from tau_eval.visualization import plot_trade_off_metric


@pytest.fixture(autouse=True)
def headless_plots(monkeypatch):
    plt.switch_backend("Agg")
    monkeypatch.setattr(plt, "show", lambda: None)
    yield
    plt.close("all")


def test_trade_off_point_matches_metric_axis_labels():
    data = {
        "dataset": {
            "selected": {"metric_x": 0.2, "metric_y": 0.8},
            "excluded": {"metric_x": 0.9, "metric_y": 0.1},
        }
    }

    plot_trade_off_metric(data, "metric_x", "metric_y", model_list=["selected"])

    axes = plt.gca()
    assert axes.get_xlabel() == "metric_x"
    assert axes.get_ylabel() == "metric_y"
    assert len(axes.collections) == 1
    assert axes.collections[0].get_label() == "selected"
    np.testing.assert_allclose(axes.collections[0].get_offsets(), [[0.2, 0.8]])


def test_trade_off_preserves_observation_pairs_and_model_groups():
    data = {
        "first_dataset": {
            "first_model": {"x": 0.2, "y": 0.8},
            "second_model": {"x": 0.9, "y": 0.1},
            "excluded_model": {"x": 0.5, "y": 0.5},
        },
        "second_dataset": {
            "first_model": {"x": 0.3, "y": "invalid"},
            "second_model": {"x": 0.4, "y": 0.6},
        },
        "third_dataset": {
            "first_model": {"x": 0.5, "y": 0.7},
            "second_model": {"y": 0.2},
        },
    }

    plot_trade_off_metric(data, "x", "y", model_list=["first_model", "second_model"])

    groups = {collection.get_label(): collection.get_offsets() for collection in plt.gca().collections}
    assert set(groups) == {"first_model", "second_model"}
    np.testing.assert_allclose(groups["first_model"], [[0.2, 0.8], [0.5, 0.7]])
    np.testing.assert_allclose(groups["second_model"], [[0.9, 0.1], [0.4, 0.6]])


def test_trade_off_preserves_task_filter():
    data = {
        "selected_task": {"model": {"x": 0.2, "y": 0.8}},
        "excluded_task": {"model": {"x": 0.9, "y": 0.1}},
    }

    plot_trade_off_metric(data, "x", "y", task_list=["selected_task"], model_list=["model"])

    np.testing.assert_allclose(plt.gca().collections[0].get_offsets(), [[0.2, 0.8]])


@pytest.mark.parametrize(
    "invalid_metrics",
    [
        {"y": 0.7},
        {"x": 0.3},
        {"x": "invalid", "y": 0.7},
        {"x": 0.3, "y": "invalid"},
        {"x": None, "y": 0.7},
        {"x": 0.3, "y": None},
        {"x": np.nan, "y": 0.7},
        {"x": 0.3, "y": np.nan},
    ],
    ids=["missing-x", "missing-y", "nonnumeric-x", "nonnumeric-y", "null-x", "null-y", "nan-x", "nan-y"],
)
def test_trade_off_filters_invalid_observations(invalid_metrics):
    data = {
        "invalid_dataset": {"model": invalid_metrics},
        "valid_dataset": {"model": {"x": 0.2, "y": 0.8}},
    }

    plot_trade_off_metric(data, "x", "y", model_list=["model"])

    assert len(plt.gca().collections) == 1
    np.testing.assert_allclose(plt.gca().collections[0].get_offsets(), [[0.2, 0.8]])


def test_trade_off_does_not_plot_when_all_observations_are_invalid(capsys):
    data = {"dataset": {"model": {"x": 0.2}, "not_metrics": "invalid"}}

    plot_trade_off_metric(data, "x", "y", model_list=["model", "not_metrics"])

    assert plt.get_fignums() == []
    assert "No text found for the selected models and metrics." in capsys.readouterr().out
