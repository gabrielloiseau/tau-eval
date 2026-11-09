import pytest

from tau_eval.tasks.deidentification import DeIdentification


class CountingText(str):
    def __new__(cls, value):
        instance = super().__new__(cls, value)
        instance.lower_calls = 0
        return instance

    def lower(self):
        self.lower_calls += 1
        return super().lower()


def _original_scores(dataset, outputs):
    counts = {}
    for example, output in zip(dataset, outputs):
        for category, values in example["labels"].items():
            if values is None:
                continue
            matched, total = counts.setdefault(category, [0, 0])
            for value in values:
                total += 1
                matched += int(value.lower() in output.lower().replace(" ", ""))
            counts[category] = [matched, total]
    result = {f"{category}_recall": matched / total for category, (matched, total) in counts.items()}
    result["Total_recall"] = sum(matched for matched, _ in counts.values()) / sum(
        total for _, total in counts.values()
    )
    return result


@pytest.mark.parametrize(
    "outputs",
    [
        ["A L I C E at New York. Ann Lee", "Bob has B@EX.COM"],
        ["No retained values", "New York and missing@ex.com"],
        ["", ""],
    ],
)
def test_deidentification_preserves_matching_counts_and_pooled_recall(outputs):
    dataset = [
        {"labels": {"NAME": ["alice", "ALICE"], "CITY": ["newyork"], "MISC": None, "MULTI": ["Ann Lee"]}},
        {"labels": {"NAME": ["bob"], "EMAIL": ["b@ex.com", "missing@ex.com"], "CITY": ["newyork"]}},
    ]

    assert DeIdentification(dataset=dataset).evaluate(outputs) == _original_scores(dataset, outputs)


def test_deidentification_normalizes_output_once_for_many_annotations():
    output = CountingText("A L I C E")
    dataset = [{"labels": {"NAME": ["Alice"] * 1000, "NULL": None}}]

    assert DeIdentification(dataset=dataset).evaluate([output]) == {"NAME_recall": 1.0, "Total_recall": 1.0}
    assert output.lower_calls == 1


def test_deidentification_normalizes_each_document_independently_without_normalizing_entity_spaces():
    outputs = [CountingText("Ann Lee"), CountingText("B O B")]
    dataset = [{"labels": {"NAME": ["Ann Lee"]}}, {"labels": {"NAME": ["Bob"]}}]

    assert DeIdentification(dataset=dataset).evaluate(outputs) == {"NAME_recall": 0.5, "Total_recall": 0.5}
    assert [output.lower_calls for output in outputs] == [1, 1]
