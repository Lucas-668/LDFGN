"""Fixed training and 72/96 search protocol for LDFGN."""

from __future__ import annotations

import csv
import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from data import PreparedDataset
from ldfgn import evaluate_ldfgn


EPOCHS = 150
LEARNING_RATE = 0.02
WEIGHT_DECAY = 1e-4
PSEUDO_RATIO = 0.8
FEATURE_REGULARIZATION = 0.0
BATCH_SIZE = 10_000

K_GRID = (1, 5)
INLIER_GRID = (0.80, 0.95, 0.975)
GAMMA_GRID = (0.5, 2.0, 3.0, 5.0)


@dataclass(frozen=True)
class Parameters:
    views: int
    K: int
    inlier: float
    gamma: float


def deterministic_base_views(original_dimensions: int) -> int:
    """Compute the label-free base view count b(D_o)."""

    if original_dimensions < 1:
        raise ValueError("original_dimensions must be positive")
    if original_dimensions < 4:
        return 1
    return max(2, min(10, original_dimensions // 3))


def view_candidates(
    original_dimensions: int,
    anchor: int,
) -> tuple[int, ...]:
    """Generate the low/base/high candidates around one anchor."""

    if original_dimensions < 1 or anchor < 1:
        raise ValueError("dimensions and anchor must be positive")
    half = int(math.floor(anchor / 2 + 0.5))
    return tuple(sorted({
        max(1, min(original_dimensions, half)),
        max(1, min(original_dimensions, anchor)),
        max(1, min(original_dimensions, 2 * anchor)),
    }))


def self_contained_view_candidates(
    original_dimensions: int,
) -> tuple[int, ...]:
    """Generate the self-contained multiscale view set from D_o only."""

    base = deterministic_base_views(original_dimensions)
    anchors = [base]
    if original_dimensions > 50:
        anchors.append(min(20, original_dimensions // 2))
    return tuple(sorted({
        view
        for anchor in anchors
        for view in view_candidates(original_dimensions, anchor)
    }))


def build_grid(original_dimensions: int) -> list[Parameters]:
    return [
        Parameters(*values)
        for values in itertools.product(
            self_contained_view_candidates(original_dimensions),
            K_GRID,
            INLIER_GRID,
            GAMMA_GRID,
        )
    ]


def compute_metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Compute direct S=1-Z metrics without label-based reversal."""

    auc = float(roc_auc_score(y, scores))
    precision, recall, _ = precision_recall_curve(y, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "auc": auc,
        "max_f1": float(np.max(f1)),
        "ap": float(average_precision_score(y, scores)),
        "orientation_failure": auc < 0.5,
    }


def train_and_score(
    dataset: PreparedDataset,
    parameters: Parameters,
    device: torch.device,
    *,
    epochs: int = EPOCHS,
) -> tuple[dict[str, object], dict[str, float]]:
    """Train one configuration with the fixed paper training profile."""

    result = evaluate_ldfgn(
        dataset,
        n_views=parameters.views,
        n_granules=parameters.K,
        inlier_ratio=parameters.inlier,
        gamma=parameters.gamma,
        device=device,
        epochs=epochs,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        pseudo_ratio=PSEUDO_RATIO,
        feature_regularization=FEATURE_REGULARIZATION,
        batch_size=BATCH_SIZE,
        view_policy="keep-blocks",
        feature_weight_mode="relu",
        loss_aggregation="sum",
        decoupled_logit_weight_decay=True,
    )
    metrics = compute_metrics(dataset.y, np.asarray(result["scores"]))
    return result, metrics


def read_parameter_rows(path: Path) -> list[dict[str, str]]:
    """Read the bundled reference table as the ordered dataset manifest."""

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "Dataset",
        "Representation",
        "Original_Dimensions",
        "Base_V",
        "Best_V",
        "Best_K",
        "Best_q",
        "Best_Gamma",
        "AUC",
        "F1",
        "AP",
    }
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise ValueError(
            f"{path} is missing columns: {', '.join(sorted(missing))}"
        )
    names = [row["Dataset"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError(f"{path} contains duplicate datasets")
    return rows


def resolve_device(name: str) -> torch.device:
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if name == "auto"
        else torch.device(name)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    return device
