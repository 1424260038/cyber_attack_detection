"""Checkpoint loading utilities for inference."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from models.cnn_lstm_model import CNNLSTMClassifier
from models.simple_classifier import SimpleClassifier
from utils.attack_taxonomy import CLASS_NAMES


def resolve_device(requested_device: str | None = None) -> torch.device:
    device_name = requested_device or os.getenv("CYBERDD_DEVICE", "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    return torch.device(device_name)


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError("Unsupported checkpoint format")


def get_checkpoint_metadata(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        return {}
    return {
        "epoch": checkpoint.get("epoch"),
        "best_val_acc": checkpoint.get("best_val_acc"),
    }


def _linear_layers(state_dict: dict[str, torch.Tensor]) -> list[tuple[int, str, torch.Tensor]]:
    layers: list[tuple[int, str, torch.Tensor]] = []
    for key, value in state_dict.items():
        if not key.endswith(".weight") or value.dim() != 2:
            continue

        parts = key.split(".")
        layer_index: int | None = None
        if parts[0].isdigit():
            layer_index = int(parts[0])
        elif len(parts) > 1 and parts[0] == "net" and parts[1].isdigit():
            layer_index = int(parts[1])

        if layer_index is not None:
            layers.append((layer_index, key, value))

    return sorted(layers, key=lambda item: item[0])


def normalize_mlp_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(key.startswith("net.") for key in state_dict):
        return state_dict
    return {f"net.{key}": value for key, value in state_dict.items()}


def _load_mlp_model(
    state_dict: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    layers = _linear_layers(state_dict)
    if len(layers) < 2:
        raise ValueError("MLP checkpoint must contain at least two linear layers")

    input_dim = int(layers[0][2].shape[1])
    hidden_dims = tuple(int(layer[2].shape[0]) for layer in layers[:-1])
    num_classes = int(layers[-1][2].shape[0])

    model = SimpleClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
    )
    model.load_state_dict(normalize_mlp_state_dict(state_dict), strict=True)
    model.to(device)
    model.eval()

    return model, {
        "architecture": "SimpleClassifier",
        "input_dim": input_dim,
        "hidden_dims": list(hidden_dims),
        "num_classes": num_classes,
        "class_names": CLASS_NAMES[:num_classes] if num_classes > 2 else ["Normal", "Attack"],
    }


def _load_cnn_lstm_model(
    state_dict: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    conv_weight = state_dict.get("conv1.conv.weight")
    classifier_weight = state_dict.get("classifier.3.weight")
    if conv_weight is None or classifier_weight is None:
        raise ValueError("Checkpoint does not look like a CNNLSTMClassifier")

    input_dim = int(conv_weight.shape[1])
    num_classes = int(classifier_weight.shape[0])

    model = CNNLSTMClassifier(input_dim=input_dim, num_classes=num_classes)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model, {
        "architecture": "CNNLSTMClassifier",
        "input_dim": input_dim,
        "num_classes": num_classes,
        "class_names": CLASS_NAMES[:num_classes] if num_classes > 2 else ["Normal", "Attack"],
    }


def load_checkpoint_model(
    checkpoint_path: str | Path,
    requested_device: str | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    device = resolve_device(requested_device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = extract_state_dict(checkpoint)

    if any(key.startswith("conv1.") for key in state_dict):
        model, info = _load_cnn_lstm_model(state_dict, device)
    else:
        model, info = _load_mlp_model(state_dict, device)

    info.update(get_checkpoint_metadata(checkpoint))
    info.update(
        {
            "checkpoint_path": str(path),
            "device": str(device),
        }
    )
    return model, info


def find_default_checkpoint(project_dir: str | Path) -> Path:
    explicit_path = os.getenv("CYBERDD_MODEL_PATH")
    if explicit_path:
        return Path(explicit_path)

    project_path = Path(project_dir)
    candidates = [
        project_path / "checkpoints" / "best_model.pth",
        project_path / "outputs" / "best_model.pth",
        project_path / "checkpoints" / "last_model.pth",
        project_path / "outputs" / "last_model.pth",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]
