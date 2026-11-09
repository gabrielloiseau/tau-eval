from unittest.mock import Mock

import pytest
import torch
from sentence_transformers.util import cos_sim

from tau_eval.metrics import luar, sbert


@pytest.fixture(params=[(sbert, "sbert"), (luar, "luar")], ids=["sbert", "luar"])
def scorer(request):
    module, key = request.param
    return module, key, getattr(module, f"compute_{key}")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_matches_installed_pairwise_reference_and_extracts_one_vector(scorer, monkeypatch, dtype):
    module, key, compute = scorer
    originals = torch.tensor([[1, 2, 3], [0, 0, 0], [1e-14, -2e-14, 0], [3, -4, 5]], dtype=dtype)
    rewrites = torch.tensor([[3, 2, 1], [0, 1, 0], [2e-14, -1e-14, 0], [-3, 4, -5]], dtype=dtype)
    expected = [cos_sim(original, rewrite).item() for original, rewrite in zip(originals, rewrites)]
    model = Mock()
    model.encode.side_effect = [originals, rewrites]
    vector_similarity = Mock(wraps=module.pairwise_cos_sim)
    monkeypatch.setattr(module, "pairwise_cos_sim", vector_similarity)
    original_tolist = torch.Tensor.tolist
    extracted_shapes = []

    def tolist(tensor):
        extracted_shapes.append(tuple(tensor.shape))
        return original_tolist(tensor)

    def forbidden(*args, **kwargs):
        pytest.fail("Neither per-item extraction nor an all-pairs matrix is needed")

    monkeypatch.setattr(torch.Tensor, "tolist", tolist)
    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    monkeypatch.setattr(torch, "mm", forbidden)
    monkeypatch.setattr(torch, "matmul", forbidden)
    inputs, outputs = ["first", "zero", "tiny", "negative"], ["one", "two", "three", "four"]

    result = compute(inputs, outputs, model)

    assert list(result) == [key]
    assert len(result[key]) == 4
    assert all(type(value) is float for value in result[key])
    assert result[key] == pytest.approx(expected, rel=1e-6, abs=1e-8)
    vector_similarity.assert_called_once_with(originals, rewrites)
    assert extracted_shapes == [(4,)]
    assert model.encode.call_args_list[0].args == (inputs,)
    assert model.encode.call_args_list[1].args == (outputs,)
    assert all(
        call.kwargs == {"convert_to_tensor": True, "show_progress_bar": False} for call in model.encode.call_args_list
    )


def test_string_inputs_return_a_one_element_list(scorer):
    _, key, compute = scorer
    model = Mock()
    model.encode.side_effect = [torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])]

    assert compute("original", "rewrite", model) == {key: [0.0]}
    assert model.encode.call_args_list[0].args == (["original"],)
    assert model.encode.call_args_list[1].args == (["rewrite"],)


def test_unequal_input_lengths_are_rejected_before_encoding(scorer):
    _, _, compute = scorer
    model = Mock()

    with pytest.raises(AssertionError, match="inputs are different lengths"):
        compute(["one", "two"], ["one"], model)
    model.encode.assert_not_called()


def test_empty_input_batches_return_empty_scores(scorer):
    _, key, compute = scorer
    assert compute([], [], Mock()) == {key: []}
