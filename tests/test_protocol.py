from __future__ import annotations

import unittest
from pathlib import Path

from protocol import (
    Parameters,
    build_grid,
    deterministic_base_views,
    read_parameter_rows,
    self_contained_view_candidates,
)


class ProtocolTests(unittest.TestCase):
    def test_label_free_base_rule(self) -> None:
        self.assertEqual(deterministic_base_views(3), 1)
        self.assertEqual(deterministic_base_views(4), 2)
        self.assertEqual(deterministic_base_views(18), 6)
        self.assertEqual(deterministic_base_views(279), 10)

    def test_view_candidates(self) -> None:
        self.assertEqual(self_contained_view_candidates(3), (1, 2))
        self.assertEqual(self_contained_view_candidates(18), (3, 6, 12))
        self.assertEqual(
            self_contained_view_candidates(279),
            (5, 10, 20, 40),
        )

    def test_grid_sizes(self) -> None:
        self.assertEqual(len(build_grid(3)), 48)
        self.assertEqual(len(build_grid(18)), 72)
        self.assertEqual(len(build_grid(279)), 96)

    def test_reference_parameters_belong_to_grid(self) -> None:
        root = Path(__file__).resolve().parents[1]
        rows = read_parameter_rows(root / "reference" / "grid_best.csv")
        self.assertEqual(len(rows), 24)
        for row in rows:
            dimensions = int(row["Original_Dimensions"])
            grid = build_grid(dimensions)
            selected = Parameters(
                int(row["Best_V"]),
                int(row["Best_K"]),
                float(row["Best_q"]),
                float(row["Best_Gamma"]),
            )
            self.assertIn(selected, grid, row["Dataset"])
            self.assertEqual(
                len(grid),
                int(row["Total_Combinations"]),
                row["Dataset"],
            )


if __name__ == "__main__":
    unittest.main()
