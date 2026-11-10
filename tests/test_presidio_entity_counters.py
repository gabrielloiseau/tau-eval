from unittest.mock import Mock

from presidio_analyzer import RecognizerResult
from presidio_anonymizer import AnonymizerEngine, OperatorConfig

from tau_eval.models import presidio


def replace(text, mapping, entity_type="PERSON"):
    # Presidio creates a separate operator instance for each recognized entity.
    operator = presidio.InstanceCounterAnonymizer()
    params = {"entity_mapping": mapping, "entity_type": entity_type}
    operator.validate(params)
    return operator.operate(text, params)


def test_entity_counters_are_contiguous_per_type_and_reuse_repeated_text():
    mapping = {}

    assert replace("Alice", mapping) == "<PERSON_0>"
    assert replace("Bob", mapping) == "<PERSON_1>"
    assert replace("Alice", mapping) == "<PERSON_0>"
    assert replace("Paris", mapping, "LOCATION") == "<LOCATION_0>"
    assert replace("Carol", mapping) == "<PERSON_2>"
    assert mapping == {
        "PERSON": {"Alice": "<PERSON_0>", "Bob": "<PERSON_1>", "Carol": "<PERSON_2>"},
        "LOCATION": {"Paris": "<LOCATION_0>"},
    }


def test_sparse_supplied_mapping_is_scanned_only_once():
    class Placeholder(str):
        scans = 0

        def split(self, *args, **kwargs):
            type(self).scans += 1
            return super().split(*args, **kwargs)

    mapping = {"PERSON": {"Alice": Placeholder("<PERSON_2>"), "Bob": Placeholder("<PERSON_8>")}}

    assert replace("Carol", mapping) == "<PERSON_9>"
    assert Placeholder.scans == 2
    for index in range(10, 30):
        assert replace(f"person-{index}", mapping) == f"<PERSON_{index}>"
    assert replace("Alice", mapping) == "<PERSON_2>"
    assert Placeholder.scans == 2


def test_existing_replacement_is_reused_without_parsing_other_values():
    mapping = {"PERSON": {"Alice": "an existing replacement"}}

    assert replace("Alice", mapping) == "an existing replacement"


def test_fresh_and_cleared_mappings_do_not_inherit_old_indices():
    mapping = {}
    assert replace("Alice", mapping) == "<PERSON_0>"
    assert replace("Bob", mapping) == "<PERSON_1>"
    assert replace("Alice", {}) == "<PERSON_0>"
    mapping["PERSON"].clear()
    assert replace("Carol", mapping) == "<PERSON_0>"
    mapping.clear()
    assert replace("Dan", mapping) == "<PERSON_0>"


def test_external_edits_have_an_explicit_reset_boundary():
    mapping = {}
    replace("Alice", mapping)
    mapping["PERSON"]["external"] = "<PERSON_12>"

    presidio.InstanceCounterAnonymizer.reset_counters(mapping)

    assert replace("Bob", mapping) == "<PERSON_13>"
    assert replace("external", mapping) == "<PERSON_12>"
    mapping["PERSON"] = {"replacement": "<PERSON_50>"}
    assert replace("Carol", mapping) == "<PERSON_51>"


def test_presidio_engine_shares_mapping_counters_across_entity_operations(monkeypatch):
    engine = AnonymizerEngine()
    engine.add_anonymizer(presidio.InstanceCounterAnonymizer)
    mapping = {"PERSON": {"existing": "<PERSON_7>"}}
    get_last_index = Mock(wraps=presidio.InstanceCounterAnonymizer._get_last_index)
    monkeypatch.setattr(presidio.InstanceCounterAnonymizer, "_get_last_index", get_last_index)
    results = [
        RecognizerResult(entity_type="PERSON", start=0, end=5, score=1),
        RecognizerResult(entity_type="PERSON", start=6, end=9, score=1),
        RecognizerResult(entity_type="PERSON", start=10, end=15, score=1),
    ]

    output = engine.anonymize(
        "Alice|Bob|Carol", results, {"DEFAULT": OperatorConfig("entity_counter", {"entity_mapping": mapping})}
    )

    assert output.text == "<PERSON_10>|<PERSON_9>|<PERSON_8>"
    assert get_last_index.call_count == 1
    assert mapping["PERSON"]["Alice"] == "<PERSON_10>"


def test_unique_placeholder_starts_a_new_mapping_per_document(monkeypatch):
    analyzer = Mock()
    analyzer.analyze.side_effect = lambda text, language: [
        RecognizerResult(entity_type="PERSON", start=0, end=len(text), score=1)
    ]
    monkeypatch.setattr(presidio, "AnalyzerEngine", Mock(return_value=analyzer))
    anonymizer = presidio.UniquePlaceholderPerEntity()

    assert anonymizer.anonymize("Alice") == "<PERSON_0>"
    assert anonymizer.anonymize("Bob") == "<PERSON_0>"
