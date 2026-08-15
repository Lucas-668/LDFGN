"""Deterministic Multi-view Adaptive Fuzzy Granular Network (LDFGN)."""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import torch
from scipy.stats import qmc
from torch import nn

from data import PreparedDataset, THETA


VIEW_METHOD = "ordered-width-balanced"
SPLIT_BLOCKS_VIEW_METHOD = "base-layout-cyclic-split-blocks"
ONE_HOT_LAYOUT_VIEW_METHOD = "one-hot-width-balanced-block-preserving"
VIEW_POLICIES = {
    "keep-blocks": VIEW_METHOD,
    "split-blocks": SPLIT_BLOCKS_VIEW_METHOD,
    "one-hot-layout": ONE_HOT_LAYOUT_VIEW_METHOD,
}
CENTER_SOBOL_START = 1
CENTER_SOBOL_CAPACITY = 16
PSEUDO_SOBOL_START = 32
PSEUDO_METHOD = "sobol-unscrambled-index-32+-collision-skip"
CENTER_METHOD = "sobol-unscrambled-index-1-16"
TIE_METHOD = "stable-score-then-sample-index"
CENTER_METHODS = {
    "sobol": CENTER_METHOD,
    "halton": "halton-unscrambled-index-1-16",
    "farthest": "deterministic-data-farthest-first-median-seed",
}
PSEUDO_METHODS = {
    "sobol": PSEUDO_METHOD,
    "halton": "halton-unscrambled-index-32+-collision-skip",
    "none": "none",
}


def configure_deterministic_execution() -> None:
    """Request deterministic kernels and disable reduced-precision shortcuts."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def make_deterministic_view_splits(
    dataset: PreparedDataset,
    n_views: int,
    *,
    view_policy: str = "keep-blocks",
) -> list[np.ndarray]:
    """Build deterministic balanced views, optionally splitting feature blocks."""

    if view_policy not in VIEW_POLICIES:
        raise ValueError(
            "view_policy must be one of: " + ", ".join(VIEW_POLICIES)
        )

    n_views = max(1, min(int(n_views), dataset.original_dimensions))
    by_feature = {block.original_index: block for block in dataset.blocks}
    ordered_features = range(dataset.original_dimensions)

    feature_assignments: list[list[int]] = [[] for _ in range(n_views)]
    encoded_load = np.zeros(n_views, dtype=np.int64)
    for feature in ordered_features:
        view = min(
            range(n_views),
            key=lambda index: (
                int(encoded_load[index]),
                len(feature_assignments[index]),
                index,
            ),
        )
        feature_assignments[view].append(feature)
        block = by_feature[feature]
        if (
            view_policy == "one-hot-layout"
            and block.kind == "categorical-integer"
        ):
            if block.category_values is None:
                raise ValueError("Integer categorical block is missing levels")
            balance_width = len(block.category_values)
        else:
            balance_width = block.width
        encoded_load[view] += balance_width

    if view_policy == "split-blocks":
        coordinate_assignments: list[list[int]] = [
            [] for _ in range(n_views)
        ]
        base_view = {
            feature: view
            for view, features in enumerate(feature_assignments)
            for feature in features
        }
        for feature in ordered_features:
            block = by_feature[feature]
            anchor = base_view[feature]
            for offset, coordinate in enumerate(block.expanded_indices):
                target = (
                    (anchor + offset) % n_views
                    if block.kind == "categorical"
                    else anchor
                )
                coordinate_assignments[target].append(int(coordinate))
        return [
            np.asarray(assignment, dtype=int)
            for assignment in coordinate_assignments
        ]

    return [
        np.concatenate(
            [by_feature[feature].expanded_indices for feature in assignment]
        ).astype(int)
        for assignment in feature_assignments
    ]


def generate_deterministic_pseudo_anomalies(
    dataset: PreparedDataset,
    n_samples: int,
    device: torch.device,
    *,
    sequence: str = "sobol",
    view_splits: Sequence[np.ndarray] | None = None,
    forbidden_centers: Sequence[np.ndarray] | None = None,
    return_rejections: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, int]:
    """Generate pseudo anomalies from a deterministic low-discrepancy block.

    Candidates start at ``PSEUDO_SOBOL_START``. When view centers are
    supplied, a candidate whose encoded projection exactly equals any center
    is deterministically discarded and replaced by the next sequence point.
    """

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if sequence not in {"sobol", "halton"}:
        raise ValueError("sequence must be 'sobol' or 'halton'")
    if (view_splits is None) != (forbidden_centers is None):
        raise ValueError(
            "view_splits and forbidden_centers must be supplied together"
        )
    if (
        view_splits is not None
        and forbidden_centers is not None
        and len(view_splits) != len(forbidden_centers)
    ):
        raise ValueError("Each view must have one forbidden center matrix")

    if sequence == "sobol":
        engine: object = torch.quasirandom.SobolEngine(
            dimension=dataset.original_dimensions,
            scramble=False,
        )
        engine.fast_forward(PSEUDO_SOBOL_START)

        def draw(count: int) -> torch.Tensor:
            return engine.draw(count, dtype=torch.float32)
    else:
        engine = qmc.Halton(
            d=dataset.original_dimensions,
            scramble=False,
        )
        engine.fast_forward(PSEUDO_SOBOL_START)

        def draw(count: int) -> torch.Tensor:
            values = engine.random(count).astype(np.float32, copy=False)
            return torch.from_numpy(values)

    accepted: list[torch.Tensor] = []
    accepted_count = 0
    rejected_count = 0
    drawn_count = 0
    while accepted_count < n_samples:
        original = draw(n_samples - accepted_count)
        drawn_count += len(original)
        pseudo_blocks: list[torch.Tensor] = []
        for block in dataset.blocks:
            coordinate = original[:, block.original_index]
            if block.kind == "categorical":
                category = torch.floor(
                    coordinate * block.width
                ).to(torch.int64)
                category.clamp_(max=block.width - 1)
                pseudo = torch.nn.functional.one_hot(
                    category, num_classes=block.width
                ).to(torch.float32)
            elif block.kind == "categorical-integer":
                if block.category_values is None:
                    raise ValueError(
                        "Integer categorical block is missing levels"
                    )
                category = torch.floor(
                    coordinate * len(block.category_values)
                ).to(torch.int64)
                category.clamp_(max=len(block.category_values) - 1)
                levels = torch.as_tensor(
                    block.category_values,
                    dtype=torch.float32,
                )
                pseudo = levels[category].unsqueeze(1)
            else:
                pseudo = coordinate.unsqueeze(1)
            pseudo_blocks.append(pseudo)
        candidates = torch.cat(pseudo_blocks, dim=1)

        collision = torch.zeros(
            len(candidates), dtype=torch.bool
        )
        if view_splits is not None and forbidden_centers is not None:
            for split, centers in zip(view_splits, forbidden_centers):
                split_index = torch.as_tensor(
                    np.asarray(split, dtype=int),
                    dtype=torch.int64,
                )
                projected = candidates[:, split_index]
                center_tensor = torch.as_tensor(
                    np.asarray(centers, dtype=np.float32)
                )
                for center in center_tensor:
                    collision |= torch.all(projected == center, dim=1)

        retained = candidates[~collision]
        accepted.append(retained)
        accepted_count += len(retained)
        rejected_count += int(torch.sum(collision).item())
        if drawn_count > max(10_000, 100 * n_samples):
            raise RuntimeError(
                "Pseudo-anomaly collision filter rejected too many "
                "candidates; the encoded support may be exhausted"
            )

    result = torch.cat(accepted, dim=0)[:n_samples].to(device)
    if return_rejections:
        return result, rejected_count
    return result


def deterministic_uniform_sobol_centers(
    view_width: int,
    n_centers: int,
) -> np.ndarray:
    """Use the reserved center block of an unscrambled Sobol sequence."""

    if view_width < 1 or n_centers < 1:
        raise ValueError("view_width and n_centers must be positive")
    if n_centers > CENTER_SOBOL_CAPACITY:
        raise ValueError(
            f"n_centers exceeds reserved Sobol capacity "
            f"{CENTER_SOBOL_CAPACITY}"
        )
    engine = torch.quasirandom.SobolEngine(
        dimension=view_width,
        scramble=False,
    )
    engine.fast_forward(CENTER_SOBOL_START)
    return engine.draw(n_centers, dtype=torch.float32).numpy().copy()


def deterministic_low_discrepancy_points(
    sequence: str,
    dimension: int,
    n_points: int,
    *,
    start: int = 0,
) -> np.ndarray:
    """Return a float64 block from an unscrambled Sobol or Halton sequence."""

    if sequence not in {"sobol", "halton"}:
        raise ValueError("sequence must be 'sobol' or 'halton'")
    if dimension < 1 or n_points < 1 or start < 0:
        raise ValueError("dimension/n_points must be positive; start >= 0")
    if sequence == "sobol":
        engine = torch.quasirandom.SobolEngine(
            dimension=dimension,
            scramble=False,
        )
        if start:
            engine.fast_forward(start)
        return engine.draw(n_points, dtype=torch.float64).numpy().copy()

    engine = qmc.Halton(d=dimension, scramble=False)
    if start:
        engine.fast_forward(start)
    return engine.random(n_points)


def deterministic_uniform_halton_centers(
    view_width: int,
    n_centers: int,
) -> np.ndarray:
    """Use indices 1..K of an unscrambled Halton sequence."""

    if view_width < 1 or n_centers < 1:
        raise ValueError("view_width and n_centers must be positive")
    if n_centers > CENTER_SOBOL_CAPACITY:
        raise ValueError(
            f"n_centers exceeds reserved center capacity "
            f"{CENTER_SOBOL_CAPACITY}"
        )
    engine = qmc.Halton(d=view_width, scramble=False)
    engine.fast_forward(CENTER_SOBOL_START)
    return engine.random(n_centers).astype(np.float32, copy=False)


def deterministic_farthest_first_centers(
    X_view: np.ndarray,
    n_centers: int,
) -> np.ndarray:
    """Select data points by a median seed and deterministic farthest-first."""

    values = np.asarray(X_view, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("X_view must be a non-empty 2-D matrix")
    if n_centers < 1 or n_centers > len(values):
        raise ValueError("n_centers must be in [1, number of samples]")

    # Accumulate distances in float64 so high-dimensional views do not lose
    # deterministic ordering through float32 summation.
    values64 = values.astype(np.float64, copy=False)
    median = np.median(values64, axis=0)
    seed_distances = np.sum((values64 - median) ** 2, axis=1)
    first = int(np.argmin(seed_distances))

    selected = [first]
    available = np.ones(len(values), dtype=bool)
    available[first] = False
    min_distances = np.sum(
        (values64 - values64[first]) ** 2,
        axis=1,
    )
    for _ in range(1, n_centers):
        candidate_distances = np.where(
            available,
            min_distances,
            -np.inf,
        )
        # np.argmax returns the lowest sample index for exact ties.
        selected_index = int(np.argmax(candidate_distances))
        selected.append(selected_index)
        available[selected_index] = False
        distances = np.sum(
            (values64 - values64[selected_index]) ** 2,
            axis=1,
        )
        min_distances = np.minimum(min_distances, distances)

    return values[np.asarray(selected, dtype=int)].copy()


def initialize_view_centers(
    dataset: PreparedDataset,
    view_splits: Sequence[np.ndarray],
    n_granules: int,
    *,
    method: str = "sobol",
) -> list[np.ndarray]:
    """Initialize every view with the selected deterministic construction."""

    if method not in CENTER_METHODS:
        raise ValueError(
            "method must be one of: " + ", ".join(CENTER_METHODS)
        )
    centers: list[np.ndarray] = []
    for split in view_splits:
        split = np.asarray(split, dtype=int)
        if method == "sobol":
            selected = deterministic_uniform_sobol_centers(
                len(split), n_granules
            )
        elif method == "halton":
            selected = deterministic_uniform_halton_centers(
                len(split), n_granules
            )
        else:
            selected = deterministic_farthest_first_centers(
                dataset.X[:, split],
                n_granules,
            )
        centers.append(selected)
    return centers


class LDFGN(nn.Module):
    """LDFGN with deterministic construction and initialization."""

    def __init__(
        self,
        n_features: int,
        n_granules: int,
        view_splits: Sequence[np.ndarray],
        initial_centers: Sequence[np.ndarray],
        theta: float = THETA,
        *,
        learn_feature_weights: bool = True,
        learn_view_weights: bool = True,
        learn_sigma: bool = True,
        feature_weight_mode: str | None = None,
    ) -> None:
        super().__init__()
        if len(view_splits) != len(initial_centers):
            raise ValueError("Each view must have one center matrix")

        if feature_weight_mode is None:
            feature_weight_mode = (
                "relu" if learn_feature_weights else "uniform"
            )
        if feature_weight_mode not in {"relu", "softmax", "uniform"}:
            raise ValueError(
                "feature_weight_mode must be 'relu', 'softmax', or "
                "'uniform'"
            )

        self.n_features = n_features
        self.n_granules = n_granules
        self.view_splits = [
            np.asarray(split, dtype=int) for split in view_splits
        ]
        self.theta = theta
        self.feature_weight_mode = feature_weight_mode
        self.learn_feature_weights = feature_weight_mode != "uniform"

        self.centers = nn.ParameterList()
        self.weights = nn.ParameterList()
        self.log_sigma2 = nn.ParameterList()
        for split, centers in zip(self.view_splits, initial_centers):
            width = len(split)
            centers = np.asarray(centers, dtype=np.float32)
            if centers.shape != (n_granules, width):
                raise ValueError(
                    "Center shape does not match (n_granules, view_width)"
                )
            self.centers.append(
                nn.Parameter(torch.from_numpy(centers.copy()))
            )
            initial_weights = (
                torch.zeros(width)
                if feature_weight_mode == "softmax"
                else torch.ones(width) / max(width, 1)
            )
            self.weights.append(
                nn.Parameter(
                    initial_weights,
                    requires_grad=self.learn_feature_weights,
                )
            )
            self.log_sigma2.append(
                nn.Parameter(
                    torch.full((width, n_granules), -0.693),
                    requires_grad=learn_sigma,
                )
            )
        self.view_alpha = nn.Parameter(
            torch.ones(len(self.view_splits)),
            requires_grad=learn_view_weights,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        view_normalities: list[torch.Tensor] = []
        for view, split in enumerate(self.view_splits):
            X_view = X[:, split]
            diff_squared = (
                X_view.unsqueeze(2) - self.centers[view].t().unsqueeze(0)
            ) ** 2
            sigma_squared = torch.exp(self.log_sigma2[view]).clamp(min=1e-3)
            similarity = torch.exp(-diff_squared / (2 * sigma_squared))

            positive_weights = self.effective_feature_weights(view)
            granule_scores = torch.sum(
                similarity * positive_weights.view(1, -1, 1), dim=1
            )
            strongest_granule = torch.max(granule_scores, dim=1).values
            normality = 1 - torch.exp(-self.theta * strongest_granule)
            view_normalities.append(normality.unsqueeze(1))

        alpha = torch.softmax(self.view_alpha, dim=0)
        return torch.sum(
            torch.cat(view_normalities, dim=1) * alpha, dim=1
        )

    def effective_feature_weights(self, view: int) -> torch.Tensor:
        if self.feature_weight_mode == "relu":
            return torch.relu(self.weights[view]) + 0.01
        if self.feature_weight_mode == "softmax":
            return torch.softmax(self.weights[view], dim=0)
        # Uniform mode fixes every within-view weight at exactly 1 / d_v.
        return self.weights[view]

    def feature_regularization(self, coefficient: float) -> torch.Tensor:
        if self.feature_weight_mode == "softmax":
            return sum(
                coefficient
                * torch.norm(self.effective_feature_weights(view)) ** 2
                for view in range(len(self.weights))
            )
        return sum(
            coefficient * torch.norm(torch.relu(weight)) ** 2
            for weight in self.weights
        )


def evaluate_ldfgn(
    dataset: PreparedDataset,
    *,
    n_views: int,
    n_granules: int,
    inlier_ratio: float,
    gamma: float,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    pseudo_ratio: float,
    feature_regularization: float,
    batch_size: int,
    center_method: str = "sobol",
    pseudo_method: str = "sobol",
    learn_feature_weights: bool = True,
    learn_view_weights: bool = True,
    trusted_partition: str = "dynamic",
    include_suspect_loss: bool = True,
    learn_sigma: bool = True,
    view_policy: str = "keep-blocks",
    feature_weight_mode: str | None = None,
    loss_aggregation: str = "sum",
    decoupled_logit_weight_decay: bool = False,
) -> dict[str, object]:
    """Train one LDFGN configuration and return its direct anomaly scores."""

    if trusted_partition not in {"dynamic", "fixed"}:
        raise ValueError("trusted_partition must be 'dynamic' or 'fixed'")
    if loss_aggregation not in {"sum", "global-mean"}:
        raise ValueError(
            "loss_aggregation must be 'sum' or 'global-mean'"
        )
    configure_deterministic_execution()
    view_splits = make_deterministic_view_splits(
        dataset,
        n_views,
        view_policy=view_policy,
    )
    initial_centers = initialize_view_centers(
        dataset,
        view_splits,
        n_granules,
        method=center_method,
    )
    model = LDFGN(
        n_features=dataset.encoded_dimensions,
        n_granules=n_granules,
        view_splits=view_splits,
        initial_centers=initial_centers,
        learn_feature_weights=learn_feature_weights,
        learn_view_weights=learn_view_weights,
        learn_sigma=learn_sigma,
        feature_weight_mode=feature_weight_mode,
    ).to(device)
    if decoupled_logit_weight_decay:
        decay_parameters = [
            parameter
            for parameter in [*model.centers, *model.log_sigma2]
            if parameter.requires_grad
        ]
        logit_parameters = [
            parameter
            for parameter in [*model.weights, model.view_alpha]
            if parameter.requires_grad
        ]
        parameter_groups = []
        if decay_parameters:
            parameter_groups.append(
                {
                    "params": decay_parameters,
                    "weight_decay": weight_decay,
                }
            )
        if logit_parameters:
            parameter_groups.append(
                {"params": logit_parameters, "weight_decay": 0.0}
            )
        optimizer = torch.optim.Adam(
            parameter_groups,
            lr=learning_rate,
        )
        optimizer_weight_decay_policy = (
            "centers-and-log-variance-only"
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        optimizer_weight_decay_policy = "all-trainable-parameters"
    X = torch.as_tensor(dataset.X, dtype=torch.float32, device=device)
    n_samples = len(dataset.X)
    if pseudo_method not in PSEUDO_METHODS:
        raise ValueError(
            "pseudo_method must be one of: " + ", ".join(PSEUDO_METHODS)
        )
    n_pseudo = (
        0 if pseudo_method == "none"
        else max(1, int(n_samples * pseudo_ratio))
    )
    pseudo: torch.Tensor | None = None
    pseudo_collision_rejections = 0
    pseudo_collision_filter = center_method != "farthest"
    if n_pseudo:
        collision_arguments: dict[str, object] = {}
        if pseudo_collision_filter:
            collision_arguments = {
                "view_splits": view_splits,
                "forbidden_centers": initial_centers,
            }
        pseudo, pseudo_collision_rejections = (
            generate_deterministic_pseudo_anomalies(
                dataset,
                n_pseudo,
                device,
                sequence=pseudo_method,
                return_rejections=True,
                **collision_arguments,
            )
        )
    trusted_count = max(1, min(n_samples, int(n_samples * inlier_ratio)))
    fixed_order: torch.Tensor | None = None
    if trusted_partition == "fixed":
        model.eval()
        with torch.no_grad():
            initial_normality = model(X)
            fixed_order = torch.argsort(
                initial_normality,
                descending=True,
                stable=True,
            )

    model.train()
    final_loss = float("nan")
    final_loss_inlier = float("nan")
    final_loss_suspect = float("nan")
    final_loss_pseudo = float("nan")
    final_loss_regularization = float("nan")
    final_loss_data = float("nan")
    first_gradient_l2 = float("nan")
    final_gradient_l2 = float("nan")
    loss_denominator = (
        float(n_samples + n_pseudo)
        if loss_aggregation == "global-mean"
        else 1.0
    )
    for epoch in range(epochs):
        optimizer.zero_grad()
        normality = model(X)
        if fixed_order is None:
            # Stable sorting resolves equal scores by the fixed input row index.
            order = torch.argsort(normality, descending=True, stable=True)
        else:
            order = fixed_order
        trusted_scores = normality[order[:trusted_count]]
        suspect_scores = normality[order[trusted_count:]]

        loss_inlier = torch.sum((1 - trusted_scores) ** 2)
        loss_suspect = (
            gamma * torch.sum(suspect_scores**2)
            if include_suspect_loss
            else normality.new_zeros(())
        )
        loss_pseudo = (
            normality.new_zeros(())
            if pseudo is None
            else gamma * torch.sum(model(pseudo) ** 2)
        )
        loss_regularization = model.feature_regularization(
            feature_regularization
        )
        loss_data = (
            loss_inlier + loss_suspect + loss_pseudo
        ) / loss_denominator
        loss = loss_data + loss_regularization
        loss.backward()
        gradient_squared = sum(
            torch.sum(parameter.grad.detach() ** 2)
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        )
        gradient_l2 = float(torch.sqrt(gradient_squared).cpu())
        if epoch == 0:
            first_gradient_l2 = gradient_l2
        if epoch == epochs - 1:
            final_gradient_l2 = gradient_l2
        optimizer.step()
        if epoch == epochs - 1:
            final_loss = float(loss.detach().cpu())
            final_loss_inlier = float(loss_inlier.detach().cpu())
            final_loss_suspect = float(loss_suspect.detach().cpu())
            final_loss_pseudo = float(loss_pseudo.detach().cpu())
            final_loss_regularization = float(
                loss_regularization.detach().cpu()
            )
            final_loss_data = float(loss_data.detach().cpu())

    model.eval()
    score_batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            normality = model(X[start : start + batch_size])
            score_batches.append((1 - normality).cpu().numpy())

    return {
        "scores": np.concatenate(score_batches),
        "view_splits": [
            split.astype(int).tolist() for split in view_splits
        ],
        "view_policy": view_policy,
        "view_method": VIEW_POLICIES[view_policy],
        "final_loss": final_loss,
        "final_loss_inlier": final_loss_inlier,
        "final_loss_suspect": final_loss_suspect,
        "final_loss_pseudo": final_loss_pseudo,
        "final_loss_regularization": final_loss_regularization,
        "final_loss_data": final_loss_data,
        "loss_aggregation": loss_aggregation,
        "loss_denominator": loss_denominator,
        "first_gradient_l2": first_gradient_l2,
        "final_gradient_l2": final_gradient_l2,
        "center_sobol_start": CENTER_SOBOL_START,
        "center_sobol_capacity": CENTER_SOBOL_CAPACITY,
        "pseudo_sobol_start": PSEUDO_SOBOL_START,
        "pseudo_collision_rejections": pseudo_collision_rejections,
        "pseudo_collision_filter": (
            "qmc-center-projection-rejection"
            if pseudo_collision_filter and n_pseudo
            else (
                "not-applicable-data-derived-centers"
                if n_pseudo
                else "not-applicable-no-pseudo"
            )
        ),
        "center_method": CENTER_METHODS[center_method],
        "pseudo_method": PSEUDO_METHODS[pseudo_method],
        "n_pseudo": n_pseudo,
        "learn_feature_weights": model.learn_feature_weights,
        "feature_weight_mode": model.feature_weight_mode,
        "final_feature_weights": [
            model.effective_feature_weights(view).detach().cpu().tolist()
            for view in range(len(model.weights))
        ],
        "final_feature_weight_sums": [
            float(
                model.effective_feature_weights(view).detach().sum().cpu()
            )
            for view in range(len(model.weights))
        ],
        "learn_view_weights": learn_view_weights,
        "trusted_partition": trusted_partition,
        "include_suspect_loss": include_suspect_loss,
        "learn_sigma": learn_sigma,
        "optimizer_weight_decay_policy": optimizer_weight_decay_policy,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }
