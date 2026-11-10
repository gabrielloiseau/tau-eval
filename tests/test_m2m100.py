import gc
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest


CHECKPOINT = "facebook/m2m100_418M"
LANGUAGES = [("en", "de"), ("de", "fr"), ("fr", "en")]


@pytest.fixture
def authorship():
    from tau_eval.models import authorship

    return authorship


@pytest.fixture
def translation_factory(authorship, monkeypatch):
    state = SimpleNamespace(loads=[], stages=[], calls=[], before_translate=None)

    def factory(task, model, *, src_lang, tgt_lang, max_new_tokens, tokenizer=None, device="cpu"):
        assert task == "translation"
        if isinstance(model, str):
            state.loads.append(model)
            tokenizer = tokenizer or model
            model = object()
        assert tokenizer == CHECKPOINT
        stage = Mock(
            model=model,
            tokenizer=SimpleNamespace(src_lang=src_lang),
            device=device,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            max_new_tokens=max_new_tokens,
        )

        def translate(texts):
            state.calls.append((src_lang, tgt_lang, list(texts)))
            if state.before_translate is not None:
                state.before_translate()
            stage.tokenizer.src_lang = src_lang
            return [{"translation_text": f"{tgt_lang}({text})"} for text in texts]

        stage.side_effect = translate
        state.stages.append(stage)
        return stage

    monkeypatch.setattr(authorship, "pipeline", factory)
    monkeypatch.setattr(authorship, "sent_tokenize", lambda text: text.split("|") if text else [])
    return state


def test_m2m100_loads_one_model_with_separate_stage_tokenizers(authorship, translation_factory):
    model = authorship.M2M100MT(name="custom name")
    stages = [model.pipeline_en_de, model.pipeline_de_fr, model.pipeline_fr_en]

    assert model.name == "custom name"
    assert translation_factory.loads == [CHECKPOINT]
    assert all(stage.model is stages[0].model for stage in stages)
    assert len({id(stage.tokenizer) for stage in stages}) == 3
    assert [(stage.src_lang, stage.tgt_lang) for stage in stages] == LANGUAGES
    assert [stage.max_new_tokens for stage in stages] == [512, 512, 512]
    stages[0].tokenizer.src_lang = "es"
    assert [stage.tokenizer.src_lang for stage in stages[1:]] == ["de", "fr"]


@pytest.mark.parametrize("batch", [False, True], ids=["single", "batch"])
def test_m2m100_preserves_sequence_and_reconstruction(authorship, translation_factory, batch):
    model = authorship.M2M100MT()

    if batch:
        assert model.anonymize_batch(["first|second", "", "third"]) == [
            "en(fr(de(first))) en(fr(de(second)))",
            "",
            "en(fr(de(third)))",
        ]
        inputs = ["first", "second", "third"]
    else:
        assert model.anonymize("first|second") == " en(fr(de(first))) en(fr(de(second)))"
        inputs = ["first", "second"]

    assert translation_factory.calls == [
        ("en", "de", inputs),
        ("de", "fr", [f"de({text})" for text in inputs]),
        ("fr", "en", [f"fr(de({text}))" for text in inputs]),
    ]


@pytest.mark.parametrize("batch", [False, True], ids=["single", "batch"])
def test_m2m100_empty_input(authorship, translation_factory, batch):
    model = authorship.M2M100MT()

    assert (model.anonymize_batch([]) if batch else model.anonymize("")) == ([] if batch else "")


def test_m2m100_serializes_overlapping_single_and_batch_calls(authorship, translation_factory):
    model = authorship.M2M100MT()
    first_entered = Event()
    second_attempted = Event()
    release_first = Event()
    lock = model._translation_lock

    class ObservedLock:
        def __enter__(self):
            if first_entered.is_set():
                second_attempted.set()
            lock.acquire()

        def __exit__(self, *args):
            lock.release()

    def pause_first_stage():
        if len(translation_factory.calls) == 1:
            first_entered.set()
            assert release_first.wait(timeout=10)

    model._translation_lock = ObservedLock()
    translation_factory.before_translate = pause_first_stage
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(model.anonymize, "first")
        try:
            assert first_entered.wait(timeout=10)
            second = executor.submit(model.anonymize_batch, ["second"])
            assert second_attempted.wait(timeout=10)
            assert len(translation_factory.calls) == 1
        finally:
            release_first.set()
        assert first.result(timeout=10) == " en(fr(de(first)))"
        assert second.result(timeout=10) == ["en(fr(de(second)))"]

    assert [(src, tgt) for src, tgt, _ in translation_factory.calls] == LANGUAGES * 2


@pytest.mark.slow
def test_m2m100_checkpoint_matches_independently_loaded_stages(authorship, monkeypatch):
    """Compare real deterministic translations; downloads the production checkpoint."""
    import torch

    from transformers import AutoModelForSeq2SeqLM, pipeline

    texts = ["The meeting starts at nine.", "Please close the window.", "Alice lives in Paris."]
    monkeypatch.setattr(authorship, "sent_tokenize", lambda text: [text] if text else [])

    def deterministic_pipeline(*args, **kwargs):
        kwargs.setdefault("device", "cpu")
        kwargs["do_sample"] = False
        return pipeline(*args, **kwargs)

    monkeypatch.setattr(authorship, "pipeline", deterministic_pipeline)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        # Each reference stage loads its own weights, as before. Release it
        # after generation to keep the integration test's peak memory bounded.
        expected_stages = []
        current = texts
        for src, tgt in LANGUAGES:
            stage = deterministic_pipeline("translation", CHECKPOINT, src_lang=src, tgt_lang=tgt, max_new_tokens=512)
            current = [item["translation_text"] for item in stage(current)]
            expected_stages.append(current)
            del stage
            gc.collect()

        with patch.object(
            AutoModelForSeq2SeqLM, "from_pretrained", wraps=AutoModelForSeq2SeqLM.from_pretrained
        ) as load:
            model = authorship.M2M100MT()
        assert load.call_count == 1
        stages = [model.pipeline_en_de, model.pipeline_de_fr, model.pipeline_fr_en]
        assert all(stage.model is stages[0].model for stage in stages)
        assert len({id(stage.tokenizer) for stage in stages}) == 3

        actual_stages = []
        original_call = type(stages[0]).__call__

        def record_stage(stage, *args, **kwargs):
            result = original_call(stage, *args, **kwargs)
            actual_stages.append([item["translation_text"] for item in result])
            return result

        with patch.object(type(stages[0]), "__call__", record_stage):
            assert model.anonymize_batch(texts) == expected_stages[-1]
        assert actual_stages == expected_stages
        assert model.anonymize(texts[0]) == " " + expected_stages[-1][0]
    finally:
        torch.set_num_threads(previous_threads)
