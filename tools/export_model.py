"""Export the current checkpoint to TorchScript for deployment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from utils.model_loader import load_checkpoint_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Export CyberDD checkpoint to TorchScript.")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--output", default="artifacts/model.pt")
    args = parser.parse_args()

    model, info = load_checkpoint_model(args.checkpoint)
    model.eval()
    example = torch.randn(1, int(info["input_dim"]))
    traced = torch.jit.trace(model, example)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output_path))
    print(f"Exported TorchScript model to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
