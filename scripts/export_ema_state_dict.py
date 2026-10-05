#!/usr/bin/env python3
"""Name UniMate's positional EMA tensors for the Rust/Burn weight loader.

This is a one-time weight-format conversion; it performs no training and does
not change tensor values. Run it in an environment with the upstream model's
Python dependencies installed (torch, torch-geometric, einops, numpy).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=repo_root / "reference" / "UniMate",
        help="path to the pinned upstream UniMate source checkout",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_root
        / "weights/unimate_uniml3d_f60_v3/checkpoints/checkpoint_step_150000.pt",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root / "weights/unimate_uniml3d_f60_v3/config.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "weights/unimate_uniml3d_f60_v3/ema_named.pt",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=repo_root / "weights/unimate_uniml3d_f60_v3/dataset_stats.npy",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=repo_root / "weights/unimate_uniml3d_f60_v3/dataset_stats.json",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.source.resolve()))
    from unimate.configs.schema import MainConfig
    from unimate.models.factory import create_model

    # weights_only=True avoids unpickling arbitrary Python objects from the
    # checkpoint. The upstream checkpoint stores tensors and primitive values.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "ema_state_dict" not in checkpoint:
        raise ValueError("checkpoint does not contain ema_state_dict")

    ema = checkpoint["ema_state_dict"]
    shadow_params = ema.get("shadow_params") if isinstance(ema, dict) else None
    if not isinstance(shadow_params, (list, tuple)):
        raise ValueError("expected ema_state_dict['shadow_params'] to be a tensor list")

    config = MainConfig.from_json(args.config)
    model = create_model(config.dataset, config.model)
    named_params = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    if len(named_params) != len(shadow_params):
        raise ValueError(
            f"EMA contains {len(shadow_params)} tensors but model has "
            f"{len(named_params)} trainable parameters"
        )

    named_ema: dict[str, torch.Tensor] = {}
    for index, ((name, parameter), shadow) in enumerate(zip(named_params, shadow_params)):
        if not isinstance(shadow, torch.Tensor):
            raise TypeError(f"EMA entry {index} ({name}) is not a tensor")
        if tuple(parameter.shape) != tuple(shadow.shape):
            raise ValueError(
                f"EMA entry {index} ({name}) has shape {tuple(shadow.shape)}; "
                f"expected {tuple(parameter.shape)}"
            )
        named_ema[name] = shadow.detach().cpu().contiguous()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(named_ema, args.output)
    print(f"Wrote {len(named_ema)} named EMA tensors to {args.output}")

    raw_stats = np.load(args.stats, allow_pickle=True).item()
    stats = _to_json_value(raw_stats)
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote normalization stats to {args.stats_output}")
    return 0


def _to_json_value(value):
    if isinstance(value, dict):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _to_json_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
