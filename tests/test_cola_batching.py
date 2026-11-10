import math

import numpy as np
import pytest
import torch

from tau_eval.metrics import cola
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer


@pytest.fixture
def inference_pair(tmp_path, monkeypatch):
    vocabulary = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "good",
        "bad",
        "very",
        "long",
        "sentence",
        "short",
        "third",
        "fourth",
        "last",
        "one",
        "brief",
    ]
    vocab_file = tmp_path / "vocab.txt"
    vocab_file.write_text("\n".join(vocabulary))
    tokenizer = BertTokenizer(vocab_file=str(vocab_file), padding_side="left")
    with torch.random.fork_rng():
        torch.manual_seed(7)
        model = BertForSequenceClassification(
            BertConfig(
                vocab_size=len(vocabulary),
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=32,
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                max_position_embeddings=64,
                num_labels=2,
                pad_token_id=tokenizer.pad_token_id,
            )
        )
    model.eval()
    monkeypatch.setattr(cola, "sent_tokenize", lambda text: text.split("|") if text.strip() else [])
    return tokenizer, model


def test_bounded_batches_match_individual_scores_and_preserve_token_ids(inference_pair, monkeypatch):
    tokenizer, model = inference_pair
    texts = ["good [PAD] sentence|bad very long sentence", "", "third|fourth brief|last one"]
    sentences = [sentence for text in texts if text for sentence in cola.sent_tokenize(text)]
    expected_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence)) for sentence in sentences]
    scores = [cola.cola_score(sentence, tokenizer, model, device="cpu") for sentence in sentences]
    expected = [np.mean(scores[:2]), np.mean(scores[2:])]
    calls = []

    def record(module, args, kwargs):
        calls.append((args[0].clone(), kwargs["attention_mask"].clone(), torch.is_grad_enabled()))

    hook = model.register_forward_pre_hook(record, with_kwargs=True)
    transfers, extractions = [], []
    original_cpu, original_tolist = torch.Tensor.cpu, torch.Tensor.tolist

    def cpu(tensor, *args, **kwargs):
        if tensor.ndim == 1 and tensor.is_floating_point():
            transfers.append(tuple(tensor.shape))
        return original_cpu(tensor, *args, **kwargs)

    def tolist(tensor):
        if tensor.ndim == 1 and tensor.is_floating_point():
            extractions.append(tuple(tensor.shape))
        return original_tolist(tensor)

    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    monkeypatch.setattr(torch.Tensor, "tolist", tolist)
    try:
        with torch.enable_grad():
            result = cola.compute_cola(texts, tokenizer, model, device="cpu", batch_size=2)
            assert torch.is_grad_enabled()
    finally:
        hook.remove()

    assert result["cola"] == pytest.approx(expected, rel=1e-5, abs=1e-6)
    assert [ids.shape[0] for ids, _, _ in calls] == [2, 2, 1]
    assert transfers == extractions == [(2,), (2,), (1,)]
    assert [grad for _, _, grad in calls] == [False, False, False]
    unpadded = [ids[row][mask[row]].tolist() for ids, mask, _ in calls for row in range(len(ids))]
    assert unpadded == expected_ids
    assert tokenizer.padding_side == "left"
    assert model.training is False
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_default_batch_limit_and_existing_positional_call(inference_pair):
    tokenizer, model = inference_pair
    sizes = []
    hook = model.register_forward_pre_hook(lambda module, args: sizes.append(len(args[0])))
    try:
        result = cola.compute_cola(["good"] * 35, tokenizer, model, "cpu")
    finally:
        hook.remove()

    assert sizes == [32, 3]
    assert len(result["cola"]) == 35
    assert all(score == pytest.approx(result["cola"][0], abs=1e-6) for score in result["cola"])


def test_single_string_and_model_training_mode_remain_supported(inference_pair):
    tokenizer, model = inference_pair
    model.train()
    expected = np.mean([cola.cola_score(text, tokenizer, model, "cpu") for text in ["good", "bad sentence"]])

    result = cola.compute_cola("good|bad sentence", tokenizer, model, "cpu", batch_size=1)

    assert result["cola"] == pytest.approx([expected], abs=1e-6)
    assert model.training is True


@pytest.mark.parametrize("texts", ["", [], ["", ""]])
def test_exact_empty_strings_are_omitted_without_inference(inference_pair, texts):
    tokenizer, model = inference_pair
    calls = []
    hook = model.register_forward_pre_hook(lambda *args: calls.append(True))
    try:
        assert cola.compute_cola(texts, tokenizer, model, "cpu") == {"cola": []}
    finally:
        hook.remove()
    assert calls == []


def test_nonempty_text_without_sentences_preserves_nan_result(inference_pair):
    tokenizer, model = inference_pair
    with pytest.warns(RuntimeWarning):
        result = cola.compute_cola(["good", " ", "bad"], tokenizer, model, "cpu")
    assert len(result["cola"]) == 3
    assert math.isnan(result["cola"][1])


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, None, True])
def test_invalid_batch_limits_are_rejected(inference_pair, batch_size):
    tokenizer, model = inference_pair
    with pytest.raises(ValueError, match="positive integer"):
        cola.compute_cola(["good"], tokenizer, model, "cpu", batch_size=batch_size)
