from unittest.mock import MagicMock, Mock

import pytest
import torch

from tau_eval.models import styleremix


ACTIVE_ADAPTERS = [
    "length_more",
    "function_more",
    "grade_more",
    "sarcasm_more",
    "formality_more",
    "voice_active",
    "type_persuasive",
]
COMBINATION = "-".join(f"{name}50" for name in ACTIVE_ADAPTERS)


@pytest.fixture
def setup(monkeypatch):
    tokenizer = MagicMock()
    tokenizer.__len__.return_value = 400
    tokenizer.eos_token_id = 2
    base_model = Mock()
    base_model.to.return_value = base_model
    adapted_model = Mock()
    adapted_model.to.return_value = adapted_model
    adapted_model.peft_config = {}

    def from_pretrained(base, path, *, adapter_name, **kwargs):
        adapted_model.peft_config[adapter_name] = path
        return adapted_model

    def load_adapter(path, *, adapter_name):
        adapted_model.peft_config[adapter_name] = path

    def add_weighted_adapter(adapters, *, weights, adapter_name, combination_type):
        adapted_model.peft_config[adapter_name] = list(adapters)

    adapted_model.load_adapter.side_effect = load_adapter
    adapted_model.add_weighted_adapter.side_effect = add_weighted_adapter
    tokenizer_loader = Mock(return_value=tokenizer)
    base_loader = Mock(return_value=base_model)
    peft_loader = Mock(side_effect=from_pretrained)
    monkeypatch.setattr(styleremix.AutoTokenizer, "from_pretrained", tokenizer_loader)
    monkeypatch.setattr(styleremix.AutoModelForCausalLM, "from_pretrained", base_loader)
    monkeypatch.setattr(styleremix.PeftModel, "from_pretrained", peft_loader)
    model = styleremix.StyleRemix()
    return model, tokenizer, base_model, adapted_model, tokenizer_loader, base_loader, peft_loader


def test_styleremix_loads_only_the_active_constituents(setup):
    model, tokenizer, base_model, adapted, tokenizer_loader, base_loader, peft_loader = setup
    first = peft_loader.call_args
    assert first.args == (base_model, styleremix.MODEL_PATHS[ACTIVE_ADAPTERS[0]])
    assert first.kwargs == {"adapter_name": ACTIVE_ADAPTERS[0], "torch_dtype": torch.float16}
    loaded = [first.kwargs["adapter_name"]] + [
        call.kwargs["adapter_name"] for call in adapted.load_adapter.call_args_list
    ]
    assert loaded == ACTIVE_ADAPTERS
    assert [call.args[0] for call in adapted.load_adapter.call_args_list] == [
        styleremix.MODEL_PATHS[name] for name in ACTIVE_ADAPTERS[1:]
    ]
    assert set(adapted.peft_config) == set(ACTIVE_ADAPTERS) | {COMBINATION}
    adapted.add_weighted_adapter.assert_called_once_with(
        ACTIVE_ADAPTERS, weights=[0.5] * 7, adapter_name=COMBINATION, combination_type="cat"
    )
    adapted.set_adapter.assert_called_once_with(COMBINATION)
    base_loader.assert_called_once_with("meta-llama/Meta-Llama-3-8B", torch_dtype=torch.float16)
    base_model.to.assert_called_once_with("cuda")
    adapted.eval.assert_called_once_with()
    assert model.device == "cuda"
    tokenizer_loader.assert_called_once_with(
        "meta-llama/Meta-Llama-3-8B", add_bos_token=True, add_eos_token=False, padding_side="left"
    )
    tokenizer.add_special_tokens.assert_called_once_with({"pad_token": "<padding_token>"})
    base_model.resize_token_embeddings.assert_called_once_with(400)


def test_styleremix_generation_uses_the_same_active_combination_and_settings(setup):
    model, tokenizer, _, adapted, *_ = setup

    class Inputs(dict):
        @property
        def input_ids(self):
            return self["input_ids"]

        def to(self, device):
            assert device == "cuda"
            return self

    inputs = Inputs(input_ids=torch.tensor([[1, 2, 3]]))
    tokenizer.return_value = inputs
    tokenizer.decode.return_value = " rewritten text \n"

    def generate(**kwargs):
        assert not torch.is_grad_enabled()
        assert kwargs["input_ids"] is inputs.input_ids
        assert kwargs["max_new_tokens"] == 1024
        assert kwargs["top_p"] == 0.95
        return torch.tensor([[1, 2, 3, 9, 8]])

    adapted.generate.side_effect = generate

    with torch.enable_grad():
        assert model.anonymize("original text") == "rewritten text"
    tokenizer.assert_called_once_with(
        "### Original: original text\n ### Rewrite:", return_tensors="pt", max_length=2048, truncation=True
    )
    assert tokenizer.pad_token_id == tokenizer.eos_token_id
    assert tokenizer.decode.call_args.args[0].tolist() == [9, 8]
    assert tokenizer.decode.call_args.kwargs == {"skip_special_tokens": True}
    adapted.set_adapter.assert_called_once_with(COMBINATION)
