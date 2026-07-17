import pytest
from safetensors.torch import save_file
from torch import nn

from api.app.v2.utils import helpers
from api.app.v2.utils.helpers import MODEL_REGISTRY, load_model


def _fake_state_dict(model_name: str) -> dict:
    build_model, head_name, _, _ = MODEL_REGISTRY[model_name]
    model = build_model()
    head = getattr(model, head_name)
    if isinstance(head, nn.Sequential):
        head[-1] = nn.Linear(head[-1].in_features, 102)
    else:
        setattr(model, head_name, nn.Linear(head.in_features, 102))
    return {k: v.contiguous() for k, v in model.state_dict().items()}


@pytest.mark.parametrize("model_name", list(MODEL_REGISTRY))
def test_load_model_selects_architecture(tmp_path, monkeypatch, model_name):
    _, _, _, hf_filename = MODEL_REGISTRY[model_name]
    weights_path = tmp_path / hf_filename
    save_file(_fake_state_dict(model_name), weights_path)

    monkeypatch.setenv("MODEL_NAME", model_name)
    monkeypatch.setattr(helpers, "hf_hub_download", lambda **kw: str(weights_path))

    model = load_model()

    assert isinstance(model, MODEL_REGISTRY[model_name][0]().__class__)


def test_load_model_rejects_unknown_model_name(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "not_a_real_model")
    with pytest.raises(ValueError, match="Unknown MODEL_NAME"):
        load_model()
