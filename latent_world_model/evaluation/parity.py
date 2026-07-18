"""Numerical parity check against VLA-JEPA's source predictor implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import torch

from ..predictor import VisionTransformerPredictorAC as StandalonePredictor


def run_parity(source_root: str | Path, checkpoint: str | Path, output: str | Path, device: str = "cuda") -> dict:
    import sys

    source = str(Path(source_root).resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    from starVLA.model.modules.world_model.vj2_predictor import VisionTransformerPredictorAC as SourcePredictor

    kwargs = {
        "num_frames": 4,
        "img_size": (256, 256),
        "tubelet_size": 1,
        "depth": 12,
        "num_heads": 8,
        "embed_dim": 2048,
        "action_embed_dim": 2048,
        "num_add_tokens": 8,
    }
    source_model = SourcePredictor(**kwargs)
    standalone_model = StandalonePredictor(**kwargs)
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True, mmap=True)
    prefix = "vj_predictor."
    predictor_state = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
    if not predictor_state:
        raise KeyError("checkpoint has no vj_predictor prefix")
    source_model.load_state_dict(predictor_state, strict=True)
    standalone_model.load_state_dict(predictor_state, strict=True)
    del state
    source_model.to(device).eval()
    standalone_model.to(device).eval()
    generator = torch.Generator(device=device).manual_seed(20260718)
    context = torch.randn((1, 768, 2048), generator=generator, device=device)
    actions = torch.randn((1, 24, 2048), generator=generator, device=device)
    with torch.inference_mode():
        source_output = source_model(context, actions)
        standalone_output = standalone_model(context, actions)
    delta = (source_output - standalone_output).abs()
    result = {
        "source_root": source,
        "checkpoint": str(Path(checkpoint).resolve()),
        "device": device,
        "input_shape": list(context.shape),
        "action_shape": list(actions.shape),
        "output_shape": list(standalone_output.shape),
        "max_abs": float(delta.max().cpu()),
        "mean_abs": float(delta.mean().cpu()),
        "allclose": bool(torch.allclose(source_output, standalone_output, rtol=0.0, atol=0.0)),
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    print(json.dumps(run_parity(args.source_root, args.checkpoint, args.output, args.device), indent=2))


if __name__ == "__main__":
    main()
