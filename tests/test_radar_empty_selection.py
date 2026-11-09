from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pytest

from tau_eval import visualization


@pytest.fixture(autouse=True)
def headless_plots(monkeypatch):
    plt.switch_backend("Agg")
    monkeypatch.setattr(plt, "show", lambda: None)
    yield
    plt.close("all")


def test_empty_model_selection_creates_no_plot_or_legend(monkeypatch):
    subplots = Mock(side_effect=AssertionError("No plot should be constructed"))
    monkeypatch.setattr(plt, "subplots", subplots)

    assert visualization.plot_radar_model_comparison({}, "score", [], ["dataset"]) is None

    subplots.assert_not_called()
    assert plt.get_fignums() == []


def test_nonempty_selection_retains_series_and_legend():
    data = {"first": {"model": {"score": 0.2}}, "second": {"model": {"score": 0.8}}}

    visualization.plot_radar_model_comparison(data, "score", ["model"], ["first", "second"])

    axes = plt.gca()
    assert len(axes.lines) == 1
    assert axes.lines[0].get_label() == "model"
    np.testing.assert_allclose(axes.lines[0].get_ydata(), [0.2, 0.8, 0.2])
    assert [text.get_text() for text in axes.get_legend().get_texts()] == ["model"]


@pytest.mark.parametrize("models", [[], ["model"]])
def test_empty_dataset_diagnostic_takes_precedence(models, capsys):
    assert visualization.plot_radar_model_comparison({}, "score", models, []) is None

    assert capsys.readouterr().out == "Error: No datasets specified for radar axes.\n"
    assert plt.get_fignums() == []
