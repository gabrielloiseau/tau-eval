import inspect

import pytest

from tau_eval.models.presidio import CategoryAnonymizer, InstanceCounterAnonymizer


@pytest.fixture(params=[InstanceCounterAnonymizer, CategoryAnonymizer])
def operator(request):
    return request.param()


def test_validate_accepts_explicit_optional_signature(operator):
    params = inspect.signature(operator.validate).parameters["params"]
    assert params.annotation == dict | None
    assert params.default is None


def test_validate_omitted_params_reports_missing_mapping(operator):
    with pytest.raises(ValueError, match="entity_mapping"):
        operator.validate()


@pytest.mark.parametrize("params", [None, {}, {"entity_type": "PERSON"}])
def test_validate_missing_mapping(operator, params):
    with pytest.raises(ValueError, match="entity_mapping"):
        operator.validate(params)


def test_validate_missing_entity_type(operator):
    with pytest.raises(ValueError, match="entity_type"):
        operator.validate({"entity_mapping": {}})


def test_validate_complete_configuration(operator):
    assert operator.validate({"entity_mapping": {}, "entity_type": "PERSON"}) is None
