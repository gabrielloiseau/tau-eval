Tutorials
=========

You can explore our tutorials to master **Tau-Eval** more effectively in the `examples/ <https://github.com/gabrielloiseau/tau-eval/tree/main/examples>`_ folder.


De-identification recall
-----------------------

``DeIdentification.evaluate`` reports the fraction of annotated sensitive
values retained in rewritten text. Each ``<category>_recall`` is the number of
matched annotated values divided by the number annotated in that category.
``Total_recall`` is pooled matched counts divided by pooled annotated counts
across all categories and documents, rather than a mean of category scores.
Repeated annotations each contribute to these counts.

Matching uses substring membership: each annotated value is lowercased and
searched for in the lowercased output after ASCII spaces (``" "``) are removed
from the output. Spaces in annotated values are not removed. Lower scores mean
more removal of annotated values under these matching rules.

For example, with two matchable annotated values:

.. code-block:: python

    from datasets import Dataset
    from tau_eval.tasks import DeIdentification

    task = DeIdentification(dataset=Dataset.from_dict({
        "text": ["Alice in Paris"],
        "labels": [{"PERSON": ["Alice"], "LOCATION": ["Paris"]}],
    }))
    task.evaluate(["Alice in Paris"])["Total_recall"]  # 1.0: both retained
    task.evaluate(["Bob in Rome"])["Total_recall"]     # 0.0: both removed
    task.evaluate(["Alice in Rome"])["Total_recall"]   # 0.5: one of two retained

In the last result, ``PERSON_recall`` is 1.0 and ``LOCATION_recall`` is 0.0.
Lower-is-better applies to removal for both category scores and ``Total_recall``.
