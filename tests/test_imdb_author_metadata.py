import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def imdb(monkeypatch):
    # Tasknet's splitting/classifier infrastructure is outside this unit test;
    # retain real pandas, NumPy selection, and Hugging Face Dataset processing.
    class Classification:
        def __init__(self, dataset, **kwargs):
            self.dataset = dataset

    tasknet = ModuleType("tasknet")
    tasknet.Classification = Classification
    tasknet.utils = SimpleNamespace(train_validation_test_split=lambda data: data)
    monkeypatch.setitem(sys.modules, "tasknet", tasknet)
    path = Path(__file__).parents[1] / "tau_eval/tasks/imdb_authorship_classification.py"
    spec = importlib.util.spec_from_file_location("imdb_metadata_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    training = {
        "userId": ["alice"] * 5 + ["bob"] * 4 + ["carol"] * 3 + ["dave"],
        "content": ["a", "b", None, "d", "e", "f", None, "h", "i", "j", "k", "l", "m"],
    }
    loader = Mock(return_value={"train": training})
    monkeypatch.setattr(module, "load_dataset", loader)
    return SimpleNamespace(module=module, training=training, loader=loader)


@pytest.mark.parametrize("seed", [0, 7, 23])
def test_author_metadata_preserves_original_selection_order_and_counts(imdb, seed):
    counts = pd.DataFrame(imdb.training)["userId"].value_counts()
    eligible = counts[counts >= 3].index
    selected = np.random.default_rng(seed).choice(np.sort(eligible), size=2, replace=False)
    expected = {
        "selected_authors": selected.tolist(),
        "author_doc_counts": {author: counts[author] for author in selected},
        "total_documents": sum(counts[author] for author in selected),
    }

    task = imdb.module.IMDBAuthorshipClassification(n_authors=2, min_docs_per_author=3, random_seed=seed)

    assert task.author_info == expected
    assert list(task.author_info["author_doc_counts"]) == selected.tolist()
    imdb.loader.assert_called_once_with("tasksource/imdb62")


def test_author_metadata_counts_rows_before_missing_value_removal(imdb):
    task = imdb.module.IMDBAuthorshipClassification(n_authors=3, min_docs_per_author=3)

    assert task.author_info["author_doc_counts"] == {"alice": 5, "bob": 4, "carol": 3}
    assert task.author_info["total_documents"] == 12
    assert len(task.dataset) == 10


def test_repeated_metadata_reads_do_not_reconstruct_data_or_resample(imdb, monkeypatch):
    task = imdb.module.IMDBAuthorshipClassification(n_authors=2, min_docs_per_author=3)
    expected = task.author_info
    forbidden = Mock(side_effect=AssertionError("metadata must not reconstruct or resample"))
    monkeypatch.setattr(imdb.module, "load_dataset", forbidden)
    monkeypatch.setattr(imdb.module.pd, "DataFrame", forbidden)
    monkeypatch.setattr(imdb.module.np.random, "default_rng", forbidden)

    for _ in range(3):
        assert task.author_info == expected
    forbidden.assert_not_called()


def test_author_metadata_remains_an_initialization_snapshot(imdb):
    task = imdb.module.IMDBAuthorshipClassification(n_authors=2, min_docs_per_author=3)
    expected = task.author_info
    task.n_authors = 1
    task.min_docs_per_author = 1
    task.random_seed = 91

    returned = task.author_info
    assert returned == expected
    returned["selected_authors"].clear()
    returned["author_doc_counts"].clear()
    returned["total_documents"] = -1
    assert task.author_info == expected
