import multiprocessing
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def _response(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _install_worker_completion(failure):
    """Install the provider mock inside each real worker, including with spawn."""
    from tau_eval.models import llm_api

    def completion(model, messages, **kwargs):
        prompt = messages[1]["content"]
        if "alice@example.com" in prompt:
            if failure == "provider":
                raise RuntimeError("Provider unavailable")
            if failure == "choices":
                return SimpleNamespace()
            if failure == "content":
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace())])
            return _response(None)
        if "first input" in prompt:
            return _response(" \nfirst anonymized\t")
        return _response("\tsecond anonymized \n")

    llm_api.completion = completion


@pytest.fixture
def llm_api(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from tau_eval.models import llm_api

    return llm_api


def test_anonymization_error_is_public_and_picklable(llm_api):
    import pickle

    from tau_eval.models import AnonymizationError

    error = pickle.loads(pickle.dumps(AnonymizationError("Anonymization failed")))
    assert type(error) is llm_api.AnonymizationError
    assert str(error) == "Anonymization failed"


def test_llm_pii_provider_failure(llm_api, monkeypatch):
    cause = RuntimeError("Provider unavailable")
    monkeypatch.setattr(llm_api, "completion", Mock(side_effect=cause))

    with pytest.raises(llm_api.AnonymizationError, match="LLM PII anonymization failed") as exc_info:
        llm_api.LLMPII().anonymize("Contact Alice at alice@example.com")

    assert exc_info.value.__cause__ is cause
    assert "alice@example.com" not in str(exc_info.value)


@pytest.mark.parametrize(
    "response",
    [
        None,
        SimpleNamespace(),
        SimpleNamespace(choices=[]),
        SimpleNamespace(choices=None),
        SimpleNamespace(choices=[SimpleNamespace()]),
        SimpleNamespace(choices=[SimpleNamespace(message=None)]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace())]),
        _response(None),
        _response(123),
        _response([]),
        _response(b"invalid content"),
    ],
    ids=[
        "null-response",
        "missing-choices",
        "empty-choices",
        "null-choices",
        "missing-message",
        "null-message",
        "missing-content",
        "null-content",
        "numeric-content",
        "list-content",
        "bytes-content",
    ],
)
def test_llm_pii_malformed_response(llm_api, monkeypatch, response):
    monkeypatch.setattr(llm_api, "completion", Mock(return_value=response))

    with pytest.raises(llm_api.AnonymizationError):
        llm_api.LLMPII().anonymize("Contact Alice at alice@example.com")


@pytest.mark.parametrize("model_name", ["test-model", "ollama/test-model"])
@pytest.mark.parametrize("content, expected", [(" \n anonymized text\t", "anonymized text"), (" \n", "")])
def test_llm_pii_success(llm_api, monkeypatch, model_name, content, expected):
    completion = Mock(return_value=_response(content))
    monkeypatch.setattr(llm_api, "completion", completion)

    assert llm_api.LLMPII(model_name).anonymize("original text", n_retries=5) == expected
    assert completion.call_args.args[0] == (model_name if "ollama" in model_name else f"openrouter/{model_name}")
    assert completion.call_args.kwargs["num_retries"] == 5


@pytest.fixture(params=["spawn", "fork"] if "fork" in multiprocessing.get_all_start_methods() else ["spawn"])
def worker_pool(request, llm_api, monkeypatch):
    def configure(failure="provider"):
        monkeypatch.setattr(
            llm_api,
            "Pool",
            partial(
                multiprocessing.get_context(request.param).Pool,
                initializer=_install_worker_completion,
                initargs=(failure,),
            ),
        )

    return configure


@pytest.mark.parametrize("failure", ["provider", "choices", "content", "null-content"])
def test_llm_pii_batch_failure_propagates(llm_api, worker_pool, failure):
    worker_pool(failure)

    with pytest.raises(llm_api.AnonymizationError, match="LLM PII anonymization failed"):
        llm_api.LLMPII().anonymize_batch(
            ["first input", "Contact Alice at alice@example.com", "second input"], n_jobs=2
        )


def test_llm_pii_batch_success_retains_order_and_strips(llm_api, worker_pool):
    worker_pool()

    assert llm_api.LLMPII().anonymize_batch(["second input", "first input"], n_jobs=2) == [
        "second anonymized",
        "first anonymized",
    ]


def test_llm_pii_empty_batch(llm_api, worker_pool):
    worker_pool()

    assert llm_api.LLMPII().anonymize_batch([], n_jobs=2) == []
