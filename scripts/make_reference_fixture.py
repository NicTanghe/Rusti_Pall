#!/usr/bin/env python3
"""Create one deterministic PyTorch forward fixture for the Burn port."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    assets = root / "weights" / "unimate_uniml3d_f60_v3"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "reference" / "UniMate")
    parser.add_argument("--config", type=Path, default=assets / "config.json")
    parser.add_argument("--weights", type=Path, default=assets / "ema_named.pt")
    parser.add_argument("--output", type=Path, default=assets / "forward_fixture.pt")
    parser.add_argument("--seed", type=int, default=7419)
    parser.add_argument("--joints", type=int, default=8, help="reduced shape for fast parity checks")
    parser.add_argument("--frames", type=int, default=4, help="reduced shape for fast parity checks")
    parser.add_argument(
        "--include-sample",
        action="store_true",
        help="also record a torchdiffeq flow sample for the Rust sampler comparison",
    )
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    args = parser.parse_args()

    sys.path.insert(0, str(args.source.resolve()))
    from unimate.configs.schema import MainConfig
    from unimate.models.factory import create_model

    config_data = json.loads(args.config.read_text(encoding="utf-8"))
    if not 1 <= args.joints <= config_data["dataset"]["max_joints"]:
        raise ValueError("--joints must be between 1 and the checkpoint maximum")
    if not 1 <= args.frames <= config_data["dataset"]["max_motion_length"]:
        raise ValueError("--frames must be between 1 and the checkpoint maximum")
    config_data["dataset"]["max_joints"] = args.joints
    config_data["dataset"]["max_motion_length"] = args.frames
    config_output = args.output.with_suffix(".config.json")
    config_output.parent.mkdir(parents=True, exist_ok=True)
    config_output.write_text(json.dumps(config_data, indent=2) + "\n", encoding="utf-8")
    config = MainConfig.from_json(config_output)
    model = create_model(config.dataset, config.model).cpu().eval()
    ema = torch.load(args.weights, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(ema, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise ValueError(f"unexpected EMA parameter names: {unexpected[:10]}")

    trace: dict[str, torch.Tensor] = {}

    def record_input_tokens(_module, _inputs, output):
        trace["trace_0"] = output[0].detach().clone()

    model.input_layer.register_forward_hook(record_input_tokens)
    for index, block in enumerate(model.transformer_blocks):
        if index == 0:
            block.register_forward_pre_hook(
                lambda _module, inputs: trace.__setitem__(
                    "trace_1", inputs[0].detach().clone()
                )
            )
        block.register_forward_hook(
            lambda _module, _inputs, output, i=index: trace.__setitem__(
                f"trace_{i + 2}", output.detach().clone()
            )
        )
    model.final_layer.register_forward_hook(
        lambda _module, _inputs, output: trace.__setitem__(
            f"trace_{len(model.transformer_blocks) + 2}", output.detach().clone()
        )
    )

    batch = 1
    joints = args.joints
    frames = args.frames
    feature_len = config.dataset.feature_len
    text_dim = 768  # google/flan-t5-base
    max_depth = config.dataset.max_depth
    max_freqs = config.model.max_freqs
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    def normal(shape):
        return torch.randn(shape, generator=generator, dtype=torch.float32)

    values = {
        "motion": normal((batch, joints, feature_len, frames)),
        "timesteps": torch.tensor([0.37], dtype=torch.float32),
        "caption_embedding": normal((batch, text_dim)),
        "tpos_first_frame": normal((batch, joints, feature_len)),
        "tpos_first_frame_parents": normal((batch, joints, feature_len)),
        "n_joints": torch.tensor([joints], dtype=torch.int64),
        "motion_lengths": torch.tensor([frames], dtype=torch.int64),
        "joint_names_emb": normal((batch, joints, text_dim)),
        "joint_depths": torch.randint(0, max_depth + 1, (batch, joints), generator=generator),
        "graph_dist": torch.randint(0, 6, (batch, joints, joints), generator=generator),
        "joint_relations": torch.randint(0, 6, (batch, joints, joints), generator=generator),
        "spectral_coords": normal((batch, joints, max_freqs)),
    }

    condition = {
        "caption_emb": values["caption_embedding"],
        "tpos_first_frame": values["tpos_first_frame"],
        "tpos_first_frame_parents": values["tpos_first_frame_parents"],
        "n_joints": values["n_joints"],
        "motion_length": values["motion_lengths"],
        "joint_names_emb": values["joint_names_emb"],
        "joint_depths": values["joint_depths"],
        "graph_dist": values["graph_dist"],
        "joint_relations": values["joint_relations"],
        "spectral_feats": values["spectral_coords"],
    }
    with torch.inference_mode():
        values["expected"] = model(
            values["motion"], values["timesteps"], cond=condition, force_mask=False
        ).contiguous()
        if args.include_sample:
            from unimate.models.factory import create_transport
            from unimate.models.flow.transport import Sampler

            class GuidedModel(torch.nn.Module):
                def __init__(self, denoiser, scale):
                    super().__init__()
                    self.denoiser = denoiser
                    self.scale = scale

                def forward(self, state, time, cond=None):
                    conditional = self.denoiser(state, time, cond=cond)
                    if self.scale <= 1.0:
                        return conditional
                    unconditional = self.denoiser(
                        state, time, cond=cond, force_mask=True
                    )
                    return unconditional + self.scale * (conditional - unconditional)

            noise = normal((batch, joints, feature_len, frames))
            transport = create_transport(training_config=config.training)
            sample_fn = Sampler(transport).sample_ode(num_steps=50)
            samples = sample_fn(noise, GuidedModel(model, args.cfg_scale), cond=condition)
            values["initial_noise"] = noise
            values["expected_sample"] = samples[-1].contiguous()
    values.update(trace)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: value.cpu().contiguous() for key, value in values.items()}, args.output)
    print(
        f"Wrote PyTorch forward fixture: output={tuple(values['expected'].shape)}, "
        f"ema_missing_aliases={len(incompatible.missing_keys)}, path={args.output}; "
        f"Burn config={config_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
