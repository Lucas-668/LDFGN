#!/usr/bin/env python3
"""Quickly reproduce the bundled best configuration for each dataset."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import platform
import time
from pathlib import Path

import torch

from data import prepare_dataset
from ldfgn import configure_deterministic_execution
from protocol import (
    EPOCHS,
    Parameters,
    build_grid,
    read_parameter_rows,
    resolve_device,
    train_and_score,
)


OUTPUT_HEADER = [
    "Dataset",
    "V",
    "K",
    "q",
    "Gamma",
    "Expected_AUC",
    "Observed_AUC",
    "AUC_Abs_Difference",
    "Expected_F1",
    "Observed_F1",
    "F1_Abs_Difference",
    "Expected_AP",
    "Observed_AP",
    "AP_Abs_Difference",
    "Parameters_In_72_96_Grid",
    "View_Splits_Exact",
    "Pseudo_Collisions_Exact",
    "Metrics_Exact",
    "All_Checks_Passed",
    "Runtime_Seconds",
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=root / "reference" / "grid_best.csv",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=root / "Datasets",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "results" / "quick_validation.csv",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--tolerance", type=float, default=5e-10)
    return parser.parse_args()


def environment(device: torch.device) -> dict[str, object]:
    gpu = None
    if device.type == "cuda":
        index = (
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        )
        properties = torch.cuda.get_device_properties(index)
        gpu = {
            "index": index,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
        }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-learn", "torch")
        },
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu": gpu,
    }


def main() -> None:
    args = parse_args()
    if args.tolerance < 0:
        raise SystemExit("--tolerance must be nonnegative")

    configure_deterministic_execution()
    reference_path = args.reference.resolve()
    data_dir = args.data_dir.resolve()
    output_path = args.output.resolve()
    reference_rows = read_parameter_rows(reference_path)
    reference = {row["Dataset"]: row for row in reference_rows}
    names = (
        list(args.datasets)
        if args.datasets
        else [row["Dataset"] for row in reference_rows]
    )
    missing = sorted(set(names) - set(reference))
    if missing:
        raise SystemExit("Unknown datasets: " + ", ".join(missing))

    device = resolve_device(args.device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_rows: list[list[object]] = []
    failures: list[str] = []
    print(
        f"LDFGN QUICK VALIDATION | device={device} | datasets={len(names)} "
        f"| epochs={EPOCHS}",
        flush=True,
    )

    for index, name in enumerate(names, start=1):
        row = reference[name]
        dataset_path = data_dir / f"{name}.mat"
        if not dataset_path.exists():
            raise FileNotFoundError(dataset_path)
        dataset = prepare_dataset(dataset_path, representation="auto")
        parameters = Parameters(
            views=int(row["Best_V"]),
            K=int(row["Best_K"]),
            inlier=float(row["Best_q"]),
            gamma=float(row["Best_Gamma"]),
        )
        expected_grid = build_grid(dataset.original_dimensions)
        parameters_in_grid = parameters in expected_grid
        expected_total = int(row["Total_Combinations"])
        grid_size_exact = len(expected_grid) == expected_total

        start = time.time()
        result, metrics = train_and_score(
            dataset,
            parameters,
            device,
            epochs=EPOCHS,
        )
        runtime = time.time() - start

        differences = {
            "auc": abs(float(metrics["auc"]) - float(row["AUC"])),
            "f1": abs(float(metrics["max_f1"]) - float(row["F1"])),
            "ap": abs(float(metrics["ap"]) - float(row["AP"])),
        }
        metrics_exact = all(
            math.isclose(value, 0.0, rel_tol=0.0, abs_tol=args.tolerance)
            for value in differences.values()
        )
        expected_splits = json.loads(row["View_Splits"])
        view_splits_exact = result["view_splits"] == expected_splits
        collisions_exact = int(result["pseudo_collision_rejections"]) == int(
            row["Pseudo_Collision_Rejections"]
        )
        passed = all((
            parameters_in_grid,
            grid_size_exact,
            metrics_exact,
            view_splits_exact,
            collisions_exact,
        ))
        if not passed:
            failures.append(name)

        output_rows.append([
            name,
            parameters.views,
            parameters.K,
            parameters.inlier,
            parameters.gamma,
            row["AUC"],
            round(float(metrics["auc"]), 10),
            differences["auc"],
            row["F1"],
            round(float(metrics["max_f1"]), 10),
            differences["f1"],
            row["AP"],
            round(float(metrics["ap"]), 10),
            differences["ap"],
            parameters_in_grid and grid_size_exact,
            view_splits_exact,
            collisions_exact,
            metrics_exact,
            passed,
            round(runtime, 3),
        ])
        print(
            f"[{index:02d}/{len(names):02d}] {name} | "
            f"AUC={float(metrics['auc']):.10f} "
            f"F1={float(metrics['max_f1']):.10f} "
            f"AP={float(metrics['ap']):.10f} | "
            f"max_delta={max(differences.values()):.3g} | "
            f"passed={passed} | time={runtime:.2f}s",
            flush=True,
        )

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(OUTPUT_HEADER)
        writer.writerows(output_rows)

    metadata = {
        "algorithm": "LDFGN",
        "experiment": "best_configuration_quick_validation",
        "reference": str(reference_path),
        "data_dir": str(data_dir),
        "epochs": EPOCHS,
        "tolerance": args.tolerance,
        "datasets": names,
        "all_checks_passed": not failures,
        "failed_datasets": failures,
        "environment": environment(device),
    }
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"RESULTS {output_path}", flush=True)
    if failures:
        raise SystemExit("Validation failed: " + ", ".join(failures))
    print("ALL CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
