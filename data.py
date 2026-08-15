"""Dataset loading and deterministic preprocessing for LDFGN."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io
from sklearn.preprocessing import MinMaxScaler


THETA = 3.0

# Zero-based categorical-column definitions for the bundled datasets.
CATEGORICAL_COLUMNS: dict[str, tuple[int, ...]] = {
    "adult_morethan50K_3779_variant1": (1, 3, 5, 6, 7, 8, 9, 13),
    "arrhythmia_variant1": tuple(
        [1]
        + [
            index
            for channel in range(12)
            for index in range(21 + 12 * channel, 27 + 12 * channel)
        ]
    ),
    "audiology_variant1": tuple(range(69)),
    "horse_1_12_variant1": tuple(
        index
        for index in range(27)
        if index not in {2, 3, 4, 5, 15, 18, 19, 21}
    ),
    "mushroom_p_221_variant1": tuple(range(22)),
    "Lymphography": tuple(range(18)),
}

REPRESENTATIONS = ("auto", "integer")


@dataclass(frozen=True)
class FeatureBlock:
    """Coordinates produced from one original attribute."""

    kind: str
    original_index: int
    width: int
    expanded_indices: np.ndarray
    category_values: np.ndarray | None = None


@dataclass(frozen=True)
class PreparedDataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    original_dimensions: int
    blocks: tuple[FeatureBlock, ...]
    representation: str

    @property
    def encoded_dimensions(self) -> int:
        return int(self.X.shape[1])


def load_mat_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a MATLAB matrix whose final column is the binary label."""

    mat = scipy.io.loadmat(path)
    keys = [key for key in mat if not key.startswith("__")]
    if not keys:
        raise ValueError(f"No data matrix found in {path}")

    data = np.asarray(mat[keys[0]])
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected a 2-D feature/label matrix in {path}")

    X = data[:, :-1].astype(np.float32)
    y_raw = np.asarray(data[:, -1]).reshape(-1)

    # In the bundled Lymphography file, label 0 denotes anomalies.
    if path.name == "Lymphography.mat":
        y = np.where(y_raw == 0, 1, 0)
    else:
        y = y_raw.astype(np.int64)

    labels = np.unique(y)
    if not np.array_equal(labels, np.array([0, 1])):
        raise ValueError(
            f"{path.name} must contain binary labels 0/1 after mapping; "
            f"found {labels.tolist()}"
        )
    return X, y


def prepare_dataset(
    path: Path,
    representation: str = "auto",
) -> PreparedDataset:
    """Min-max scale numerical attributes and one-hot encode categoricals."""

    if representation not in REPRESENTATIONS:
        raise ValueError(
            "representation must be one of: " + ", ".join(REPRESENTATIONS)
        )

    X_raw, y = load_mat_dataset(path)
    name = path.stem
    categorical = set(CATEGORICAL_COLUMNS.get(name, ()))
    if categorical and max(categorical) >= X_raw.shape[1]:
        raise ValueError(
            f"Categorical-column definition for {name} exceeds its "
            f"{X_raw.shape[1]} input columns"
        )
    if representation == "integer" and not categorical:
        raise ValueError(
            f"{name} has no declared categorical columns for integer coding"
        )

    transformed: list[np.ndarray] = []
    blocks: list[FeatureBlock] = []
    offset = 0
    for feature in range(X_raw.shape[1]):
        if feature in categorical:
            values = np.unique(X_raw[:, feature])
            lookup = {
                value: index for index, value in enumerate(values.tolist())
            }
            codes = np.asarray(
                [lookup[value] for value in X_raw[:, feature]],
                dtype=np.int64,
            )
            if representation == "integer":
                block = MinMaxScaler().fit_transform(
                    X_raw[:, [feature]]
                ).astype(np.float32)
                np.clip(block, 0.0, 1.0, out=block)
                category_values = np.asarray(
                    [
                        block[codes == index, 0][0]
                        for index in range(len(values))
                    ],
                    dtype=np.float32,
                )
                width = 1
                kind = "categorical-integer"
            else:
                width = len(values)
                block = np.eye(width, dtype=np.float32)[codes]
                category_values = np.arange(width, dtype=np.float32)
                kind = "categorical"
        else:
            block = MinMaxScaler().fit_transform(
                X_raw[:, [feature]]
            ).astype(np.float32)
            width = 1
            kind = "numerical"
            category_values = None

        indices = np.arange(offset, offset + width, dtype=int)
        transformed.append(block)
        blocks.append(
            FeatureBlock(
                kind=kind,
                original_index=feature,
                width=width,
                expanded_indices=indices,
                category_values=category_values,
            )
        )
        offset += width

    output_representation = (
        "integer"
        if representation == "integer"
        else ("one-hot" if categorical else "minmax")
    )
    return PreparedDataset(
        name=name,
        X=np.concatenate(transformed, axis=1).astype(np.float32),
        y=y,
        original_dimensions=X_raw.shape[1],
        blocks=tuple(blocks),
        representation=output_representation,
    )
