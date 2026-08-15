#!/usr/bin/env python3
"""Resumable deterministic LDFGN grid search with live progress logging."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import platform
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import roc_curve

from ldfgn import (
    CENTER_METHOD,
    CENTER_SOBOL_CAPACITY,
    CENTER_SOBOL_START,
    PSEUDO_METHOD,
    PSEUDO_SOBOL_START,
    TIE_METHOD,
    VIEW_POLICIES,
    configure_deterministic_execution,
    evaluate_ldfgn,
)
from data import (
    CATEGORICAL_COLUMNS,
    REPRESENTATIONS,
    PreparedDataset,
    prepare_dataset,
)
from protocol import read_parameter_rows
from protocol import (
    BATCH_SIZE,
    EPOCHS,
    FEATURE_REGULARIZATION,
    LEARNING_RATE,
    PSEUDO_RATIO,
    WEIGHT_DECAY,
    Parameters,
    compute_metrics,
    deterministic_base_views,
    self_contained_view_candidates,
)


K_GRID = (1, 5)
INLIER_GRID = (0.80, 0.95, 0.975)
GAMMA_GRID = (0.5, 2.0, 3.0, 5.0)
GRID_PROFILES = {
    "auc-72-96": (K_GRID, INLIER_GRID, GAMMA_GRID),
}
VIEW_GRID_RULES = ("self-contained-multiscale",)
ROC_POINT_COUNT = 101
TRAINING_PROFILES = ("A",)

DETAIL_HEADER = [
    "Dataset",
    "Representation",
    "Combo_Index",
    "Total_Combinations",
    "Samples",
    "Original_Dimensions",
    "Encoded_Dimensions",
    "Base_V",
    "V",
    "K",
    "q",
    "Gamma",
    "AUC",
    "F1",
    "AP",
    "Orientation_Failure",
    "Runtime_Seconds",
    "Final_Loss",
    "Center_Sobol_Start",
    "Center_Sobol_Capacity",
    "Pseudo_Sobol_Start",
    "Pseudo_Collision_Rejections",
    "View_Method",
    "Pseudo_Method",
    "Center_Method",
    "Tie_Method",
    "View_Splits",
]

BEST_HEADER = [
    "Dataset",
    "Representation",
    "Samples",
    "Original_Dimensions",
    "Encoded_Dimensions",
    "Base_V",
    "Best_V",
    "Best_K",
    "Best_q",
    "Best_Gamma",
    "AUC",
    "F1",
    "AP",
    "Orientation_Failure",
    "Completed_Combinations",
    "Total_Combinations",
    "Runtime_Seconds",
    "Center_Sobol_Start",
    "Center_Sobol_Capacity",
    "Pseudo_Sobol_Start",
    "Pseudo_Collision_Rejections",
    "View_Method",
    "Pseudo_Method",
    "Center_Method",
    "Tie_Method",
    "View_Splits",
]

ROC_HEADER = [
    "Dataset",
    "Point_Index",
    "FPR",
    "TPR",
    "Best_V",
    "Best_K",
    "Best_q",
    "Best_Gamma",
    "Exact_AUC",
]

RAW_ROC_HEADER = [
    "Dataset",
    "Point_Index",
    "FPR",
    "TPR",
    "Threshold",
    "Best_V",
    "Best_K",
    "Best_q",
    "Best_Gamma",
    "Exact_AUC",
]


def experiment_view_candidates(
    original_dimensions: int,
    v0: int,
    fixed_views: int | None = None,
    *,
    view_grid_rule: str = "self-contained-multiscale",
) -> tuple[int, ...]:
    if view_grid_rule not in VIEW_GRID_RULES:
        raise ValueError(
            "view_grid_rule must be one of: " + ", ".join(VIEW_GRID_RULES)
        )
    if fixed_views is not None and fixed_views < 1:
        raise ValueError("fixed_views must be positive")
    if fixed_views is not None:
        return (min(original_dimensions, fixed_views),)
    return self_contained_view_candidates(original_dimensions)


def build_ldfgn_grid(
    original_dimensions: int,
    v0: int,
    fixed_views: int | None = None,
    *,
    view_grid_rule: str = "self-contained-multiscale",
    K_grid: tuple[int, ...] = K_GRID,
    inlier_grid: tuple[float, ...] = INLIER_GRID,
    gamma_grid: tuple[float, ...] = GAMMA_GRID,
) -> list[Parameters]:
    from itertools import product

    return [Parameters(*values) for values in product(
        experiment_view_candidates(
            original_dimensions,
            v0,
            fixed_views,
            view_grid_rule=view_grid_rule,
        ),
        K_grid,
        inlier_grid,
        gamma_grid,
    )]


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    return device


def initialize_csv(path: Path, header: list[str], overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not path.exists():
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(header)


def append_row(path: Path, row: Iterable[object]) -> None:
    # Opening per row intentionally makes every completed combination durable.
    with path.open("a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(row)
        handle.flush()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_reference(
    generated_path: Path,
    reference_path: Path,
    dataset_names: list[str],
    *,
    tolerance: float = 5e-10,
) -> None:
    """Require generated best parameters and metrics to match the reference."""

    generated = {row["Dataset"]: row for row in read_rows(generated_path)}
    reference = {
        row["Dataset"]: row for row in read_parameter_rows(reference_path)
    }
    exact_columns = (
        "Representation",
        "Samples",
        "Original_Dimensions",
        "Encoded_Dimensions",
        "Base_V",
        "Best_V",
        "Best_K",
        "Best_q",
        "Best_Gamma",
        "Orientation_Failure",
        "Completed_Combinations",
        "Total_Combinations",
        "Center_Sobol_Start",
        "Center_Sobol_Capacity",
        "Pseudo_Sobol_Start",
        "Pseudo_Collision_Rejections",
        "View_Method",
        "Pseudo_Method",
        "Center_Method",
        "Tie_Method",
        "View_Splits",
    )
    metric_columns = ("AUC", "F1", "AP")
    errors: list[str] = []
    for name in dataset_names:
        if name not in generated or name not in reference:
            errors.append(f"{name}: missing generated/reference row")
            continue
        current = generated[name]
        expected = reference[name]
        for column in exact_columns:
            if current[column] != expected[column]:
                errors.append(
                    f"{name}/{column}: {current[column]!r} != "
                    f"{expected[column]!r}"
                )
        for column in metric_columns:
            if not math.isclose(
                float(current[column]),
                float(expected[column]),
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                errors.append(
                    f"{name}/{column}: {current[column]} != {expected[column]}"
                )
    if errors:
        raise RuntimeError(
            "Reference verification failed:\n" + "\n".join(errors[:30])
        )
    print(
        f"REFERENCE CHECK PASSED | datasets={len(dataset_names)} | "
        f"tolerance={tolerance:g}",
        flush=True,
    )


def replace_dataset_rows(
    path: Path,
    header: list[str],
    dataset_name: str,
    rows: Iterable[Iterable[object]],
) -> None:
    """Atomically replace one dataset's rows in a resumable output CSV."""

    retained = [
        row
        for row in read_rows(path)
        if row.get("Dataset") != dataset_name
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in retained:
            writer.writerow([row[column] for column in header])
        writer.writerows(rows)
        handle.flush()
    temporary.replace(path)


def fixed_fpr_roc(
    labels: np.ndarray,
    scores: np.ndarray,
    n_points: int = ROC_POINT_COUNT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return raw ROC values and an ROC interpolated at fixed FPR values."""

    raw_fpr, raw_tpr, thresholds = roc_curve(
        labels,
        scores,
        drop_intermediate=False,
    )
    unique_fpr = np.unique(raw_fpr)
    upper_tpr = np.array(
        [np.max(raw_tpr[raw_fpr == value]) for value in unique_fpr]
    )
    grid_fpr = np.linspace(0.0, 1.0, n_points)
    grid_tpr = np.interp(grid_fpr, unique_fpr, upper_tpr)
    # Keep the conventional plotted ROC endpoints. The raw CSV retains every
    # vertical segment and should be used when exact curve geometry matters.
    grid_tpr[0] = 0.0
    grid_tpr[-1] = 1.0
    return raw_fpr, raw_tpr, thresholds, grid_fpr, grid_tpr


def record_from_detail(row: dict[str, str]) -> dict[str, object]:
    return {
        "views": int(row["V"]),
        "K": int(row["K"]),
        "inlier": float(row["q"]),
        "gamma": float(row["Gamma"]),
        "auc": float(row["AUC"]),
        "f1": float(row["F1"]),
        "ap": float(row["AP"]),
        "orientation_failure": row["Orientation_Failure"].lower() == "true",
        "runtime": float(row["Runtime_Seconds"]),
        "final_loss": float(row["Final_Loss"]),
        "pseudo_collision_rejections": int(
            row["Pseudo_Collision_Rejections"]
        ),
        "view_splits": json.loads(row["View_Splits"]),
    }


def record_from_best(row: dict[str, str]) -> dict[str, object]:
    return {
        "views": int(row["Best_V"]),
        "K": int(row["Best_K"]),
        "inlier": float(row["Best_q"]),
        "gamma": float(row["Best_Gamma"]),
        "auc": float(row["AUC"]),
        "f1": float(row["F1"]),
        "ap": float(row["AP"]),
        "orientation_failure": row["Orientation_Failure"].lower() == "true",
        "runtime": float(row["Runtime_Seconds"]),
        "pseudo_collision_rejections": int(
            row["Pseudo_Collision_Rejections"]
        ),
        "view_splits": json.loads(row["View_Splits"]),
    }


def is_better_auc(
    candidate: dict[str, object],
    incumbent: dict[str, object] | None,
) -> bool:
    if incumbent is None:
        return True
    candidate_key = (
        float(candidate["auc"]),
        float(candidate["f1"]),
        float(candidate["ap"]),
    )
    incumbent_key = (
        float(incumbent["auc"]),
        float(incumbent["f1"]),
        float(incumbent["ap"]),
    )
    return candidate_key > incumbent_key


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def evaluate_record(
    dataset: PreparedDataset,
    record: dict[str, object],
    device: torch.device,
    epochs: int,
    view_policy: str = "keep-blocks",
    training_profile: str = "A",
) -> tuple[dict[str, object], dict[str, float]]:
    if training_profile not in TRAINING_PROFILES:
        raise ValueError(
            "training_profile must be one of: "
            + ", ".join(TRAINING_PROFILES)
        )
    profile_arguments = {
        "feature_regularization": FEATURE_REGULARIZATION,
        "feature_weight_mode": "relu",
        "loss_aggregation": "sum",
        "decoupled_logit_weight_decay": True,
    }
    result = evaluate_ldfgn(
        dataset,
        n_views=int(record["views"]),
        n_granules=int(record["K"]),
        inlier_ratio=float(record["inlier"]),
        gamma=float(record["gamma"]),
        device=device,
        epochs=epochs,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        pseudo_ratio=PSEUDO_RATIO,
        batch_size=BATCH_SIZE,
        view_policy=view_policy,
        **profile_arguments,
    )
    metrics = compute_metrics(dataset.y, np.asarray(result["scores"]))
    return result, metrics


def export_best_roc(
    *,
    dataset: PreparedDataset,
    best: dict[str, object],
    output_dir: Path,
    device: torch.device,
    epochs: int,
    view_policy: str = "keep-blocks",
    training_profile: str = "A",
) -> None:
    """Deterministically rerun the best configuration and export its ROC."""

    rerun_start = time.time()
    result, metrics = evaluate_record(
        dataset,
        best,
        device,
        epochs,
        view_policy,
        training_profile,
    )
    if not math.isclose(
        float(metrics["auc"]),
        float(best["auc"]),
        rel_tol=0.0,
        abs_tol=5e-10,
    ):
        raise RuntimeError(
            f"{dataset.name}: best-configuration rerun AUC changed from "
            f"{float(best['auc']):.10f} to {float(metrics['auc']):.10f}"
        )
    if int(result["pseudo_collision_rejections"]) != int(
        best["pseudo_collision_rejections"]
    ):
        raise RuntimeError(
            f"{dataset.name}: pseudo collision count changed on ROC rerun"
        )

    raw_fpr, raw_tpr, thresholds, grid_fpr, grid_tpr = fixed_fpr_roc(
        dataset.y,
        np.asarray(result["scores"]),
    )
    exact_auc = round(float(metrics["auc"]), 10)
    parameters = [
        int(best["views"]),
        int(best["K"]),
        float(best["inlier"]),
        float(best["gamma"]),
        exact_auc,
    ]
    grid_rows = [
        [
            dataset.name,
            index,
            round(float(fpr), 10),
            round(float(tpr), 10),
            *parameters,
        ]
        for index, (fpr, tpr) in enumerate(zip(grid_fpr, grid_tpr))
    ]
    raw_rows = [
        [
            dataset.name,
            index,
            round(float(fpr), 10),
            round(float(tpr), 10),
            float(threshold),
            *parameters,
        ]
        for index, (fpr, tpr, threshold) in enumerate(
            zip(raw_fpr, raw_tpr, thresholds)
        )
    ]
    replace_dataset_rows(
        output_dir / "grid_best_roc_101.csv",
        ROC_HEADER,
        dataset.name,
        grid_rows,
    )
    replace_dataset_rows(
        output_dir / "grid_best_roc_raw.csv",
        RAW_ROC_HEADER,
        dataset.name,
        raw_rows,
    )
    print(
        f"ROC  [{dataset.name}] points={len(grid_rows)} "
        f"raw_points={len(raw_rows)} AUC={exact_auc:.10f} "
        f"rerun={format_duration(time.time() - rerun_start)}",
        flush=True,
    )


def write_wide_roc(output_dir: Path, dataset_names: list[str]) -> None:
    """Write one shared-FPR table for datasets with complete ROC exports."""

    rows = read_rows(output_dir / "grid_best_roc_101.csv")
    by_dataset: dict[str, list[dict[str, str]]] = {}
    for name in dataset_names:
        selected = [row for row in rows if row["Dataset"] == name]
        selected.sort(key=lambda row: int(row["Point_Index"]))
        if len(selected) == ROC_POINT_COUNT:
            by_dataset[name] = selected
    if not by_dataset:
        return

    names = list(by_dataset)
    path = output_dir / "grid_best_roc_101_wide.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["Point_Index", "FPR", *[f"{name}_TPR" for name in names]]
        )
        for index in range(ROC_POINT_COUNT):
            writer.writerow([
                index,
                by_dataset[names[0]][index]["FPR"],
                *[by_dataset[name][index]["TPR"] for name in names],
            ])


def run_dataset(
    *,
    dataset: PreparedDataset,
    dataset_index: int,
    dataset_count: int,
    v0: int,
    output_dir: Path,
    device: torch.device,
    epochs: int,
    max_combinations: int | None,
    log_every: int,
    fixed_views: int | None,
    view_policy: str,
    training_profile: str,
    view_grid_rule: str,
    K_grid: tuple[int, ...],
    inlier_grid: tuple[float, ...],
    gamma_grid: tuple[float, ...],
) -> None:
    detail_path = output_dir / "grid_detail.csv"
    best_path = output_dir / "grid_best.csv"
    roc_path = output_dir / "grid_best_roc_101.csv"
    raw_roc_path = output_dir / "grid_best_roc_raw.csv"
    views = experiment_view_candidates(
        dataset.original_dimensions,
        v0,
        fixed_views,
        view_grid_rule=view_grid_rule,
    )
    grid = build_ldfgn_grid(
        dataset.original_dimensions,
        v0,
        fixed_views,
        view_grid_rule=view_grid_rule,
        K_grid=K_grid,
        inlier_grid=inlier_grid,
        gamma_grid=gamma_grid,
    )
    if max_combinations is not None:
        grid = grid[:max_combinations]

    completed_best = {
        row["Dataset"]: row for row in read_rows(best_path)
    }
    roc_counts = Counter(row["Dataset"] for row in read_rows(roc_path))
    raw_roc_counts = Counter(
        row["Dataset"] for row in read_rows(raw_roc_path)
    )
    if (
        dataset.name in completed_best
        and roc_counts[dataset.name] == ROC_POINT_COUNT
        and raw_roc_counts[dataset.name] > 1
    ):
        print(
            f"SKIP [{dataset_index}/{dataset_count}] {dataset.name}: "
            "best configuration and ROC outputs already complete",
            flush=True,
        )
        return
    if dataset.name in completed_best:
        print(
            f"RESUME ROC [{dataset_index}/{dataset_count}] {dataset.name}: "
            "grid is complete; regenerating ROC outputs",
            flush=True,
        )
        export_best_roc(
            dataset=dataset,
            best=record_from_best(completed_best[dataset.name]),
            output_dir=output_dir,
            device=device,
            epochs=epochs,
            view_policy=view_policy,
            training_profile=training_profile,
        )
        return

    existing_rows = [
        row
        for row in read_rows(detail_path)
        if row["Dataset"] == dataset.name
    ]
    completed = {int(row["Combo_Index"]) for row in existing_rows}
    previous_runtime = sum(
        float(row["Runtime_Seconds"]) for row in existing_rows
    )
    best: dict[str, object] | None = None
    for row in existing_rows:
        record = record_from_detail(row)
        if is_better_auc(record, best):
            best = record

    print(
        f"\nDATASET [{dataset_index}/{dataset_count}] {dataset.name} | "
        f"N={len(dataset.X)} | D={dataset.original_dimensions}"
        f"->{dataset.encoded_dimensions} | V_base={v0} | "
        f"V_rule={view_grid_rule} | "
        f"V={list(views)} | "
        f"combinations={len(grid)} | resumed={len(completed)}",
        flush=True,
    )

    dataset_start = time.time()
    newly_completed = 0
    for combo_index, params in enumerate(grid):
        if combo_index in completed:
            continue

        combo_start = time.time()
        result, metrics = evaluate_record(
            dataset,
            asdict(params),
            device,
            epochs,
            view_policy,
            training_profile,
        )
        runtime = time.time() - combo_start
        record = {
            **asdict(params),
            "auc": float(metrics["auc"]),
            "f1": float(metrics["max_f1"]),
            "ap": float(metrics["ap"]),
            "orientation_failure": bool(metrics["orientation_failure"]),
            "runtime": runtime,
            "final_loss": float(result["final_loss"]),
            "pseudo_collision_rejections": int(
                result["pseudo_collision_rejections"]
            ),
            "view_splits": result["view_splits"],
        }
        if is_better_auc(record, best):
            best = record

        append_row(detail_path, [
            dataset.name,
            dataset.representation,
            combo_index,
            len(grid),
            len(dataset.X),
            dataset.original_dimensions,
            dataset.encoded_dimensions,
            v0,
            params.views,
            params.K,
            params.inlier,
            params.gamma,
            round(float(metrics["auc"]), 10),
            round(float(metrics["max_f1"]), 10),
            round(float(metrics["ap"]), 10),
            metrics["orientation_failure"],
            round(runtime, 3),
            round(float(result["final_loss"]), 10),
            result["center_sobol_start"],
            result["center_sobol_capacity"],
            result["pseudo_sobol_start"],
            result["pseudo_collision_rejections"],
            result["view_method"],
            PSEUDO_METHOD,
            CENTER_METHOD,
            TIE_METHOD,
            json.dumps(result["view_splits"], separators=(",", ":")),
        ])

        newly_completed += 1
        finished = len(completed) + newly_completed
        elapsed = previous_runtime + (time.time() - dataset_start)
        mean_runtime = elapsed / max(finished, 1)
        eta = mean_runtime * (len(grid) - finished)
        if (
            finished % log_every == 0
            or finished == len(grid)
            or newly_completed == 1
        ):
            print(
                f"[{dataset_index:02d}/{dataset_count:02d}] "
                f"{dataset.name} [{finished:03d}/{len(grid):03d}] "
                f"V={params.views:>2} K={params.K:>2} "
                f"q={params.inlier:.4f} gamma={params.gamma:g} | "
                f"AUC={float(metrics['auc']):.4f} "
                f"F1={float(metrics['max_f1']):.4f} "
                f"AP={float(metrics['ap']):.4f} | "
                f"best={float(best['auc']):.4f} | "
                f"time={runtime:.2f}s eta={format_duration(eta)}",
                flush=True,
            )

    if best is None:
        raise RuntimeError(f"No combinations available for {dataset.name}")

    total_runtime = previous_runtime + (time.time() - dataset_start)
    append_row(best_path, [
        dataset.name,
        dataset.representation,
        len(dataset.X),
        dataset.original_dimensions,
        dataset.encoded_dimensions,
        v0,
        best["views"],
        best["K"],
        best["inlier"],
        best["gamma"],
        round(float(best["auc"]), 10),
        round(float(best["f1"]), 10),
        round(float(best["ap"]), 10),
        best["orientation_failure"],
        len(grid),
        len(grid),
        round(total_runtime, 3),
        CENTER_SOBOL_START,
        CENTER_SOBOL_CAPACITY,
        PSEUDO_SOBOL_START,
        best["pseudo_collision_rejections"],
        VIEW_POLICIES[view_policy],
        PSEUDO_METHOD,
        CENTER_METHOD,
        TIE_METHOD,
        json.dumps(best["view_splits"], separators=(",", ":")),
    ])
    print(
        f"DONE [{dataset_index}/{dataset_count}] {dataset.name} | "
        f"best V={best['views']} K={best['K']} q={best['inlier']} "
        f"gamma={best['gamma']} | AUC={float(best['auc']):.4f} "
        f"F1={float(best['f1']):.4f} AP={float(best['ap']):.4f} | "
        f"runtime={format_duration(total_runtime)}",
        flush=True,
    )
    export_best_roc(
        dataset=dataset,
        best=best,
        output_dir=output_dir,
        device=device,
        epochs=epochs,
        view_policy=view_policy,
        training_profile=training_profile,
    )


def write_metadata(
    output_dir: Path,
    *,
    device: torch.device,
    epochs: int,
    dataset_names: list[str],
    parameter_path: Path,
    max_combinations: int | None,
    shard_index: int,
    num_shards: int,
    fixed_views: int | None,
    representation: str,
    view_policy: str,
    training_profile: str,
    view_grid_rule: str,
    grid_profile: str,
    K_grid: tuple[int, ...],
    inlier_grid: tuple[float, ...],
    gamma_grid: tuple[float, ...],
) -> None:
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
    metadata = {
        "algorithm": "LDFGN",
        "experiment": "LDFGN_self_contained_72_96_grid_search",
        "score_definition": "S = 1 - Z",
        "score_reversal_using_labels": False,
        "f1_definition": "maximum F1 over precision-recall thresholds",
        "selection": "best full-dataset labeled AUC; oracle benchmark",
        "roc_export": {
            "points_per_dataset": ROC_POINT_COUNT,
            "fixed_grid": "FPR = 0.00, 0.01, ..., 1.00",
            "interpolation": "upper TPR at duplicate FPR, then linear",
            "raw_points_exported": True,
            "best_configuration_rerun": True,
        },
        "grid": {
            "profile": grid_profile,
            "view_grid_rule": view_grid_rule,
            "V": (
                "union over c in B(D_o) of "
                "{round_half_up(c/2), c, min(D_o, 2c)}; "
                "B(D_o)={b(D_o)} for D_o<=50 and "
                "B(D_o)={b(D_o), min(20, floor(D_o/2))} for D_o>50"
            ),
            "computed_base_formula": (
                "b(D_o)=1 for D_o<4; otherwise "
                "max(2,min(10,floor(D_o/3)))"
            ),
            "detail_csv_base_column": "Base_V",
            "detail_csv_base_semantics": "computed b(D_o)",
            "K": list(K_grid),
            "q": list(inlier_grid),
            "gamma": list(gamma_grid),
        },
        "deterministic_components": {
            "views": VIEW_POLICIES[view_policy],
            "pseudo_anomalies": PSEUDO_METHOD,
            "centers": CENTER_METHOD,
            "trusted_ties": TIE_METHOD,
            "kernels": "torch deterministic algorithms; TF32 disabled",
            "sobol_intervals": {
                "center_start_inclusive": CENTER_SOBOL_START,
                "center_capacity": CENTER_SOBOL_CAPACITY,
                "pseudo_start_inclusive": PSEUDO_SOBOL_START,
                "post_encoding_collision_policy": (
                    "deterministically reject and draw next Sobol point"
                ),
            },
        },
        "training": {
            "profile": training_profile,
            "epochs": epochs,
            "learning_rate": LEARNING_RATE,
            "center_log_variance_weight_decay": WEIGHT_DECAY,
            "feature_view_weight_decay": 0.0,
            "pseudo_ratio": PSEUDO_RATIO,
            "feature_regularization": FEATURE_REGULARIZATION,
            "feature_weight_mode": "relu-positive-unnormalized",
            "loss_aggregation": "sum",
            "batch_size": BATCH_SIZE,
        },
        "datasets": dataset_names,
        "one_hot_datasets": sorted(CATEGORICAL_COLUMNS),
        "representation_request": representation,
        "view_policy": view_policy,
        "parameters": str(parameter_path),
        "max_combinations": max_combinations,
        "fixed_views": fixed_views,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "device": str(device),
        "gpu": gpu,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "scikit-learn", "torch")
            },
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        },
    }
    with (output_dir / "grid_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)


def parse_args() -> argparse.Namespace:
    code_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Resumable LDFGN grid search with live CSV logging."
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=code_dir / "reference" / "grid_best.csv",
        help="Reference table and ordered dataset manifest.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=code_dir / "Datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            code_dir
            / "results"
            / "full_grid"
        ),
    )
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--training-profile",
        choices=TRAINING_PROFILES,
        default="A",
        help="Fixed paper training profile.",
    )
    parser.add_argument(
        "--grid-profile",
        choices=tuple(GRID_PROFILES),
        default="auc-72-96",
        help="Fixed 72/96 grid profile.",
    )
    parser.add_argument(
        "--view-grid-rule",
        choices=VIEW_GRID_RULES,
        default="self-contained-multiscale",
        help="Compute candidates from original dimension D_o only.",
    )
    parser.add_argument(
        "--representation",
        choices=REPRESENTATIONS,
        default="auto",
        help="auto keeps the established preprocessing; integer uses one scaled coordinate per categorical attribute.",
    )
    parser.add_argument(
        "--view-policy",
        choices=tuple(VIEW_POLICIES),
        default="keep-blocks",
        help="Keep one-hot attribute blocks intact or distribute encoded coordinates independently.",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--fixed-views",
        type=int,
        default=None,
        help="Use one fixed V for every dataset instead of the V0 grid.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=None,
        help="Debug only: truncate each dataset grid.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print live progress every N completed combinations.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing CSV files instead of resuming.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs must be positive")
    if args.log_every < 1:
        raise SystemExit("--log-every must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")
    if args.max_combinations is not None and args.max_combinations < 1:
        raise SystemExit("--max-combinations must be positive")
    if args.fixed_views is not None and args.fixed_views < 1:
        raise SystemExit("--fixed-views must be positive")

    configure_deterministic_execution()
    parameter_path = args.parameters.resolve()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    K_grid, inlier_grid, gamma_grid = GRID_PROFILES[args.grid_profile]
    parameter_rows = read_parameter_rows(parameter_path)
    by_name = {row["Dataset"]: row for row in parameter_rows}

    if args.datasets:
        missing = sorted(set(args.datasets) - set(by_name))
        if missing:
            raise SystemExit(f"Unknown datasets: {', '.join(missing)}")
        names = list(args.datasets)
    elif (
        args.representation == "integer"
        or args.view_policy == "split-blocks"
    ):
        names = [
            row["Dataset"]
            for row in parameter_rows
            if row["Dataset"] in CATEGORICAL_COLUMNS
        ]
    else:
        names = [row["Dataset"] for row in parameter_rows]
    names = names[args.shard_index :: args.num_shards]
    if not names:
        raise SystemExit("No datasets selected for this shard")

    device = resolve_device(args.device)
    output_dir.mkdir(parents=True, exist_ok=True)
    initialize_csv(
        output_dir / "grid_detail.csv",
        DETAIL_HEADER,
        args.overwrite,
    )
    initialize_csv(
        output_dir / "grid_best.csv",
        BEST_HEADER,
        args.overwrite,
    )
    initialize_csv(
        output_dir / "grid_best_roc_101.csv",
        ROC_HEADER,
        args.overwrite,
    )
    initialize_csv(
        output_dir / "grid_best_roc_raw.csv",
        RAW_ROC_HEADER,
        args.overwrite,
    )
    write_metadata(
        output_dir,
        device=device,
        epochs=args.epochs,
        dataset_names=names,
        parameter_path=parameter_path,
        max_combinations=args.max_combinations,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        fixed_views=args.fixed_views,
        representation=args.representation,
        view_policy=args.view_policy,
        training_profile=args.training_profile,
        view_grid_rule=args.view_grid_rule,
        grid_profile=args.grid_profile,
        K_grid=K_grid,
        inlier_grid=inlier_grid,
        gamma_grid=gamma_grid,
    )

    print(
        f"LDFGN GRID START | device={device} | datasets={len(names)} | "
        f"epochs={args.epochs} | profile={args.training_profile} | "
        f"grid_profile={args.grid_profile} | "
        f"view_rule={args.view_grid_rule} | "
        f"resume={not args.overwrite}",
        flush=True,
    )
    print(
        f"K={list(K_grid)} | q={list(inlier_grid)} | "
        f"gamma={list(gamma_grid)}",
        flush=True,
    )
    print(f"OUTPUT {output_dir}", flush=True)

    run_start = time.time()
    for dataset_index, name in enumerate(names, start=1):
        dataset_path = data_dir / f"{name}.mat"
        if not dataset_path.exists():
            raise FileNotFoundError(dataset_path)
        dataset = prepare_dataset(
            dataset_path,
            representation=args.representation,
        )
        v0 = deterministic_base_views(dataset.original_dimensions)
        run_dataset(
            dataset=dataset,
            dataset_index=dataset_index,
            dataset_count=len(names),
            v0=v0,
            output_dir=output_dir,
            device=device,
            epochs=args.epochs,
            max_combinations=args.max_combinations,
            log_every=args.log_every,
            fixed_views=args.fixed_views,
            view_policy=args.view_policy,
            training_profile=args.training_profile,
            view_grid_rule=args.view_grid_rule,
            K_grid=K_grid,
            inlier_grid=inlier_grid,
            gamma_grid=gamma_grid,
        )

    write_wide_roc(output_dir, names)
    if args.max_combinations is None and args.fixed_views is None:
        verify_reference(
            output_dir / "grid_best.csv",
            parameter_path,
            names,
        )
    print(
        f"LDFGN GRID COMPLETE | datasets={len(names)} | "
        f"wall_time={format_duration(time.time() - run_start)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
