from unittest.mock import Mock

import pytest

from tau_eval.models import authorship


@pytest.fixture(params=[authorship.KeepItSimple, authorship.Paraphraser, authorship.M2M100MT])
def model(request, monkeypatch):
    monkeypatch.setattr(authorship, "sent_tokenize", lambda text: text.split("|") if text else [])

    def factory(*args, **kwargs):
        if kwargs.get("model") == "philippelaban/keep_it_simple":
            return Mock(
                side_effect=lambda texts: [[{"generated_text": text + f"rewritten:{text[:-13]}"}] for text in texts]
            )
        if args and args[0] == "translation":
            target = kwargs["tgt_lang"]
            return Mock(side_effect=lambda texts: [{"translation_text": f"{target}({text})"} for text in texts])
        return Mock(side_effect=lambda texts: [{"generated_text": f"rewritten:{text}"} for text in texts])

    monkeypatch.setattr(authorship, "pipeline", factory)
    return request.param()


@pytest.mark.parametrize(
    "texts, groups",
    [
        (
            ["first|second", "", "third", "fourth|fifth|sixth"],
            [["first", "second"], [], ["third"], ["fourth", "fifth", "sixth"]],
        ),
        (["", "first", ""], [[], ["first"], []]),
        (["", ""], [[], []]),
        ([], []),
    ],
    ids=["varying-lengths", "leading-and-trailing-empty", "all-empty", "empty-batch"],
)
def test_authorship_reconstruction_preserves_document_order_and_separators(model, texts, groups):
    transform = (
        (lambda text: f"en(fr(de({text})))")
        if isinstance(model, authorship.M2M100MT)
        else (lambda text: f"rewritten:{text}")
    )
    expected = [" ".join(transform(sentence) for sentence in group) for group in groups]

    assert model.anonymize_batch(texts) == expected
    flattened = [sentence for group in groups for sentence in group]
    if isinstance(model, authorship.KeepItSimple):
        model.pipeline.assert_called_once_with([text + "<|endoftext|>" for text in flattened])
    elif isinstance(model, authorship.Paraphraser):
        model.pipeline.assert_called_once_with(flattened)
    else:
        model.pipeline_en_de.assert_called_once_with(flattened)
        model.pipeline_de_fr.assert_called_once_with([f"de({text})" for text in flattened])
        model.pipeline_fr_en.assert_called_once_with([f"fr(de({text}))" for text in flattened])
