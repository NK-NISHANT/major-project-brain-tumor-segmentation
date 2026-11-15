"""Unit tests for 3D Volumetric HD95 metrics and edge-case handling.

Workspace: D:\\MajorProject
Target: src/evaluation/metrics.py
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from src.evaluation.metrics import (
    CompartmentHD95,
    VolumeHD95Metrics,
    compute_compartment_hd95,
    compute_volume_hd95,
)


def _create_synthetic_sphere(
    shape: tuple[int, int, int] = (64, 64, 64),
    center: tuple[float, float, float] = (32.0, 32.0, 32.0),
    radius: float = 12.0,
) -> np.ndarray:
    """Generate a binary 3D sphere mask."""
    x = np.arange(shape[0], dtype=np.float32)
    y = np.arange(shape[1], dtype=np.float32)
    z = np.arange(shape[2], dtype=np.float32)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    dist_sq = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (zz - center[2]) ** 2
    return (dist_sq <= radius**2).astype(np.uint8)


class TestHD95Metrics(unittest.TestCase):
    def test_identical_synthetic_spheres_zero_distance(self) -> None:
        """Identical ground truth and prediction must yield HD95 == 0.0 mm."""
        mask = _create_synthetic_sphere()
        res = compute_compartment_hd95(mask, mask, spacing=(1.0, 1.0, 1.0))
        self.assertEqual(res.status, "valid")
        self.assertAlmostEqual(res.raw_hd95, 0.0, places=2)
        self.assertAlmostEqual(res.bounded_hd95, 0.0, places=2)

    def test_translated_synthetic_spheres_distance(self) -> None:
        """Translating a sphere by delta_x should yield HD95 close to translation distance."""
        sphere1 = _create_synthetic_sphere(center=(30.0, 32.0, 32.0), radius=10.0)
        sphere2 = _create_synthetic_sphere(center=(34.0, 32.0, 32.0), radius=10.0)
        res = compute_compartment_hd95(sphere1, sphere2, spacing=(1.0, 1.0, 1.0))
        self.assertEqual(res.status, "valid")
        # Translation of 4.0 mm along X axis
        self.assertGreater(res.raw_hd95, 3.5)
        self.assertLess(res.raw_hd95, 4.5)

    def test_anisotropic_spacing_scaling(self) -> None:
        """Testing anisotropic spacing scales distances in physical millimeters."""
        sphere1 = _create_synthetic_sphere(center=(32.0, 32.0, 28.0), radius=8.0)
        sphere2 = _create_synthetic_sphere(center=(32.0, 32.0, 32.0), radius=8.0)
        # Translation is 4 slices along Z axis.
        # With isotropic spacing (1, 1, 1), distance ~ 4.0 mm
        res_iso = compute_compartment_hd95(sphere1, sphere2, spacing=(1.0, 1.0, 1.0))
        # With anisotropic spacing (1, 1, 2.5), distance along Z scales up significantly
        res_aniso = compute_compartment_hd95(sphere1, sphere2, spacing=(1.0, 1.0, 2.5))
        self.assertGreater(res_aniso.raw_hd95, res_iso.raw_hd95 * 1.8)

    def test_empty_mask_edge_cases(self) -> None:
        """All three empty-mask combinations must be handled safely without exceptions."""
        empty = np.zeros((40, 40, 40), dtype=np.uint8)
        non_empty = _create_synthetic_sphere(shape=(40, 40, 40), center=(20, 20, 20), radius=8.0)

        # 1. Both empty: True Negative agreement
        res_both_empty = compute_compartment_hd95(empty, empty, max_penalty=373.12)
        self.assertEqual(res_both_empty.status, "both_empty")
        self.assertTrue(math.isnan(res_both_empty.raw_hd95))
        self.assertEqual(res_both_empty.bounded_hd95, 0.0)

        # 2. Complete miss: Prediction empty, GT present
        res_miss = compute_compartment_hd95(empty, non_empty, max_penalty=373.12)
        self.assertEqual(res_miss.status, "complete_miss")
        self.assertTrue(math.isinf(res_miss.raw_hd95))
        self.assertEqual(res_miss.bounded_hd95, 373.12)

        # 3. False alarm: Prediction present, GT empty
        res_false_alarm = compute_compartment_hd95(non_empty, empty, max_penalty=373.12)
        self.assertEqual(res_false_alarm.status, "false_alarm")
        self.assertTrue(math.isinf(res_false_alarm.raw_hd95))
        self.assertEqual(res_false_alarm.bounded_hd95, 373.12)

    def test_volume_hierarchical_compartments_wt_tc_et(self) -> None:
        """Volume HD95 correctly decomposes multiclass volume into WT, TC, ET."""
        # Class 1: Necrotic (NCR), Class 2: Edema (ED), Class 3: Enhancing (ET)
        vol_gt = np.zeros((48, 48, 48), dtype=np.int64)
        vol_pred = np.zeros((48, 48, 48), dtype=np.int64)

        # Edema region (class 2)
        vol_gt[15:35, 15:35, 15:35] = 2
        vol_pred[15:35, 15:35, 15:35] = 2

        # Tumor Core - Necrotic (class 1)
        vol_gt[20:30, 20:30, 20:30] = 1
        vol_pred[20:30, 20:30, 20:30] = 1

        # Enhancing Tumor (class 3) present in GT but omitted in prediction
        vol_gt[22:28, 22:28, 22:28] = 3
        # vol_pred leaves class 3 as class 1 (missing ET)

        metrics = compute_volume_hd95(vol_pred, vol_gt, spacing=(1.0, 1.0, 1.0))
        self.assertIsInstance(metrics, VolumeHD95Metrics)
        # WT is perfectly aligned (classes 1+2+3 all map to WT)
        self.assertEqual(metrics.wt.status, "valid")
        self.assertAlmostEqual(metrics.wt.raw_hd95, 0.0, places=2)

        # TC is perfectly aligned (classes 1+3 all map to TC)
        self.assertEqual(metrics.tc.status, "valid")
        self.assertAlmostEqual(metrics.tc.raw_hd95, 0.0, places=2)

        # ET is a complete miss (GT has class 3, pred has no class 3)
        self.assertEqual(metrics.et.status, "complete_miss")
        self.assertTrue(math.isinf(metrics.et.raw_hd95))
        self.assertEqual(metrics.et.bounded_hd95, 373.12)

    def test_torch_tensor_and_numpy_interoperability(self) -> None:
        """Verify inputs as PyTorch tensors or numpy arrays produce identical results."""
        sphere_np = _create_synthetic_sphere()
        sphere_torch = torch.from_numpy(sphere_np)

        res_np = compute_compartment_hd95(sphere_np, sphere_np)
        res_torch = compute_compartment_hd95(sphere_torch, sphere_torch)

        self.assertEqual(res_np.status, res_torch.status)
        self.assertAlmostEqual(res_np.raw_hd95, res_torch.raw_hd95, places=3)


if __name__ == "__main__":
    unittest.main()
