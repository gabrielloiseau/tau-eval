from unittest.mock import Mock

import pytest
import torch

from tau_eval.metrics import cola
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class RecordingClassifier(PreTrainedModel):
    def __init__(self):
        super().__init__(PretrainedConfig())
        self.class_logits = torch.nn.Parameter(torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
        self.grad_modes = []
        self.logits_require_grad = []
        self.inputs = []
        self.failure = None

    def forward(self, input_ids, attention_mask=None):
        self.grad_modes.append(torch.is_grad_enabled())
        self.inputs.append(input_ids.detach().clone())
        if self.failure is not None:
            raise self.failure
        logits = self.class_logits[input_ids[:, 0]]
        self.logits_require_grad.append(logits.requires_grad)
        return SequenceClassifierOutput(logits=logits)


@pytest.fixture
def classifier():
    return RecordingClassifier()


@pytest.fixture
def tokenizer():
    tokenizer = Mock()
    tokenizer.pad_token_id = 0
    tokenizer.tokenize.side_effect = str.split
    tokenizer.convert_tokens_to_ids.side_effect = lambda tokens: [int(token == "good") for token in tokens]
    return tokenizer


@pytest.mark.parametrize("caller_grad", [True, False], ids=["grad-enabled", "grad-disabled"])
@pytest.mark.parametrize("training", [True, False], ids=["train-mode", "eval-mode"])
def test_cola_score_disables_gradients_without_changing_score_or_model_mode(
    classifier, tokenizer, caller_grad, training
):
    classifier.train(training)
    expected = classifier.class_logits.detach()[1].softmax(-1)[1].item()
    original_grad_mode = torch.is_grad_enabled()
    assert all(parameter.requires_grad for parameter in classifier.parameters())

    with torch.set_grad_enabled(caller_grad):
        result = cola.cola_score("good sentence", tokenizer, classifier, device="cpu")
        assert torch.is_grad_enabled() is caller_grad

    assert torch.is_grad_enabled() is original_grad_mode
    assert type(result) is float
    assert result == expected
    assert classifier.grad_modes == [False]
    assert classifier.logits_require_grad == [False]
    assert classifier.training is training
    assert classifier.inputs[0].tolist() == [[1, 0]]
    tokenizer.tokenize.assert_called_once_with("good sentence")
    tokenizer.convert_tokens_to_ids.assert_called_once_with(["good", "sentence"])


@pytest.mark.parametrize("caller_grad", [True, False], ids=["grad-enabled", "grad-disabled"])
def test_cola_score_restores_grad_mode_after_classifier_exception(classifier, tokenizer, caller_grad):
    classifier.failure = RuntimeError("classifier failed")
    original_grad_mode = torch.is_grad_enabled()

    with torch.set_grad_enabled(caller_grad):
        with pytest.raises(RuntimeError, match="classifier failed") as exc_info:
            cola.cola_score("good sentence", tokenizer, classifier, device="cpu")
        assert torch.is_grad_enabled() is caller_grad

    assert exc_info.value is classifier.failure
    assert torch.is_grad_enabled() is original_grad_mode
    assert classifier.grad_modes == [False]
    assert classifier.training is True


def test_cola_score_probability_extraction_also_disables_gradients(classifier, tokenizer, monkeypatch):
    observed = []
    original_softmax = torch.Tensor.softmax
    original_item = torch.Tensor.item

    def softmax(tensor, *args, **kwargs):
        observed.append(("softmax", torch.is_grad_enabled()))
        return original_softmax(tensor, *args, **kwargs)

    def item(tensor, *args, **kwargs):
        observed.append(("item", torch.is_grad_enabled()))
        return original_item(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "softmax", softmax)
    monkeypatch.setattr(torch.Tensor, "item", item)

    with torch.enable_grad():
        cola.cola_score("good sentence", tokenizer, classifier, device="cpu")
        assert torch.is_grad_enabled()

    assert observed == [("softmax", False), ("item", False)]


def test_compute_cola_preserves_sentence_means_and_skips_empty_texts(classifier, tokenizer, monkeypatch):
    monkeypatch.setattr(cola, "sent_tokenize", lambda text: text.split("|"))
    probabilities = classifier.class_logits.detach().softmax(-1)[:, 1].tolist()

    with torch.enable_grad():
        result = cola.compute_cola(
            ["good sentence|bad sentence", "", "good sentence", ""], tokenizer, classifier, device="cpu"
        )
        assert torch.is_grad_enabled()

    assert result["cola"] == pytest.approx([(probabilities[1] + probabilities[0]) / 2, probabilities[1]])
    assert classifier.grad_modes and not any(classifier.grad_modes)
    assert classifier.logits_require_grad and not any(classifier.logits_require_grad)


def test_compute_cola_accepts_a_single_string(classifier, tokenizer, monkeypatch):
    monkeypatch.setattr(cola, "sent_tokenize", lambda text: [text])
    expected = classifier.class_logits.detach()[0].softmax(-1)[1].item()

    assert cola.compute_cola("bad sentence", tokenizer, classifier, device="cpu")["cola"] == [expected]


@pytest.mark.parametrize("texts", ["", [], ["", ""]], ids=["empty-string", "empty-list", "empty-documents"])
def test_compute_cola_empty_text_behavior(classifier, tokenizer, texts):
    assert cola.compute_cola(texts, tokenizer, classifier, device="cpu") == {"cola": []}
    assert classifier.inputs == []
    tokenizer.tokenize.assert_not_called()
