import torch.nn as nn
from typing import Any, Dict, Literal

from models_new import STraTS, SurpriseSTraTS, SurpriseSTraTS_VT


def _merge_cfg(base_cfg: Dict[str, Any], **overrides: Any) -> Dict[str, Any]:
    merged = dict(base_cfg)
    for key, value in overrides.items():
        if isinstance(value, dict):
            merged[key] = {**value, **dict(merged.get(key) or {})}
        else:
            merged.setdefault(key, value)
    return merged


def build_model(
    name: Literal["strats", "surprise", "surprise_vt", "surprise_vttg"],
    cfg: Dict[str, Any],
    num_features: int,
    n_output: int,
) -> nn.Module:
    cfg = dict(cfg or {})

    if name == "strats":
        return STraTS(num_features=num_features, n_output=n_output, **cfg)
    if name == "surprise":
        cfg = _merge_cfg(
            cfg,
            surprise_args={"sim_threshold": 0.90, "direction": "past"},
        )
        return SurpriseSTraTS(num_features=num_features, n_output=n_output, **cfg)
    if name == "surprise_vt":
        cfg = _merge_cfg(
            cfg,
            vt_mask_args={"sim_threshold": 1.40, "direction": "past"},
        )
        return SurpriseSTraTS_VT(num_features=num_features, n_output=n_output, **cfg)
    if name == "surprise_vttg":
        cfg = _merge_cfg(
            cfg,
            use_surprise=False,
            use_timegap_surprise=True,
            vt_mask_args={"sim_threshold": 0.80, "direction": "past"},
        )
        return SurpriseSTraTS_VT(num_features=num_features, n_output=n_output, **cfg)
    raise ValueError(f"Unknown model_name: {name}")
