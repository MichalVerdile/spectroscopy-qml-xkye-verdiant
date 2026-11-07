#!/usr/bin/env python3
"""Original parameter census (meta device) and native forward/backward profiling.

This creates disposable models; no training checkpoint or RNG state is reused.
Legacy time/gradient/GPU claims require --execute-legacy on adequate hardware.
"""

import argparse
import importlib
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from spectroscopy_qml.benchmarking.raw_models import build_model
from spectroscopy_qml.benchmarking.raw_native import NATIVE_LENGTHS, atomic_json
from spectroscopy_qml.benchmarking.raw_training import seed_everything, sync


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--execute-legacy", action="store_true")
    parser.add_argument("--parameter-limit", type=int, default=100_000_000)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    rows = []
    for modality in ("cnmr", "hnmr", "ir", "msms_pos", "msms_neg"):
        config_module = importlib.import_module(
            f"spectroscopy_qml.mps_classifier.{modality}.config"
        )
        original_config = asdict(config_module.ModelConfig())
        model_module = importlib.import_module(f"spectroscopy_qml.mps_classifier.{modality}.model")
        with torch.device("meta"):
            original = model_module.MPSFunctionalGroupClassifier(**original_config)
        count = sum(p.numel() for p in original.parameters())
        row = {
            "model": "mps_legacy",
            "modality": modality,
            "config": original_config,
            "parameter_count": count,
            "float32_weight_bytes": count * 4,
            "adamw_parameter_gradient_moments_bytes_lower_bound": count * 16,
            "forward_backward_seconds": None,
            "gradient_norm": None,
            "gpu_peak_bytes": None,
            "note": "Meta-device exact parameter census; no legacy runtime or GPU measurement",
        }
        del original
        if args.execute_legacy:
            if count > args.parameter_limit:
                raise ValueError(
                    f"Legacy model requires {count} parameters; raise explicit limit on suitable hardware"
                )
            seed_everything(42)
            original = model_module.MPSFunctionalGroupClassifier(**original_config).to(args.device)
            x = torch.randn(2, original_config["input_dim"], device=args.device)
            sync(args.device)
            started = time.perf_counter()
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            original(x).square().mean().backward()
            sync(args.device)
            row.update(
                forward_backward_seconds=time.perf_counter() - started,
                gradient_norm=float(
                    torch.nn.utils.clip_grad_norm_(
                        original.parameters(), float("inf"), error_if_nonfinite=True
                    )
                ),
                gpu_peak_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
                note="Measured disposable historical model; synthetic batch=2, no optimizer update",
            )
            del original
        rows.append(row)
        length, channels = NATIVE_LENGTHS.get(modality, 31), 3 if modality.startswith("msms") else 2
        for family in ("cnn", "mps", "ttn"):
            seed_everything(42)
            model = build_model(family, length, channels, {}).to(args.device)
            x = torch.randn(2, length, channels, device=args.device)
            x[..., -1] = 1
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            sync(args.device)
            started = time.perf_counter()
            model(x).square().mean().backward()
            sync(args.device)
            rows.append(
                {
                    "model": model.architecture,
                    "modality": modality,
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                    "forward_backward_seconds": time.perf_counter() - started,
                    "gradient_norm": float(
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), float("inf"), error_if_nonfinite=True
                        )
                    ),
                    "gpu_peak_bytes": torch.cuda.max_memory_allocated()
                    if args.device == "cuda"
                    else None,
                    "input_length": length,
                    "channels_including_mask": channels,
                }
            )
            del model
    atomic_json(
        args.output,
        {
            "device": args.device,
            "batch_size": 2,
            "threads": args.threads,
            "timing": "single forward/backward, no warmup, diagnostic only",
            "rows": rows,
        },
    )
    print(f"Profiled {len(rows)} entries to {args.output}")


if __name__ == "__main__":
    main()
