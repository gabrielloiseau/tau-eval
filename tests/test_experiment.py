
import json
from datasets import Dataset
from tau_eval.experiment import Experiment, ExperimentConfig
from tau_eval.models.dummy import DummyModel
from tau_eval.tasks.customtask import CustomTask

def test_experiment_run(tmp_path):
    # Create a dummy task
    class DummyTask(CustomTask):
        def __init__(self):
            self.name = "dummy_task"
            self.dataset = Dataset.from_dict({"text": ["hello world"]})

        def evaluate(self, new_texts):
            return {"dummy_metric": 1.0}

    # Configure and run the experiment
    models = [DummyModel()]
    metrics = ["rouge"]
    tasks = [DummyTask()]
    config = ExperimentConfig()
    exp = Experiment(models, metrics, tasks, config)
    output_file = tmp_path / "test_results.json"
    exp.run(output_file)

    # Check that the results file was created
    assert output_file.exists()

    # Check the content of the results file
    with output_file.open() as f:
        results = json.load(f)
        assert "dummy_task_0" in results
        task_results = results["dummy_task_0"]
        assert "Dummy Model" in task_results
        assert "rouge1" in task_results["Dummy Model"]
        assert task_results["original_metrics"] == {"dummy_metric": 1.0}
        assert task_results["Dummy Model"]["dummy_metric"] == 1.0
