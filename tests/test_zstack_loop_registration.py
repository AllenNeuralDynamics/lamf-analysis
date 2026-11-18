import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import tifffile

from lamf_analysis.ophys.zstack import (
    _load_registered_loop_volumes,
    reg_between_planes,
    summarize_volume_shifts,
)


def _truth(rng, n_planes, n_channels, size):
    """Smooth random 'cells', distinct per plane and channel, in a positive intensity range."""
    img = ndi.gaussian_filter(rng.random((n_planes, n_channels, size, size)), (0, 0, 2, 2))
    img -= img.min(axis=(-2, -1), keepdims=True)
    img /= img.max(axis=(-2, -1), keepdims=True)
    return 200 + 2000 * img


def _write_loop_tiff(path, truth, n_volumes, motion, noise_sd, rng):
    """Write pages in ScanImage loop order (volume, plane, channel); each frame rolled by motion[v, z]."""
    n_planes, n_channels, size, _ = truth.shape
    frames = np.empty((n_volumes, n_planes, n_channels, size, size), dtype=np.int16)
    for v in range(n_volumes):
        for z in range(n_planes):
            moved = np.roll(truth[z], tuple(motion[v, z]), axis=(-2, -1))
            frames[v, z] = np.clip(moved + rng.normal(0, noise_sd, moved.shape), 0, 32000)
    tifffile.imwrite(path, frames.reshape(-1, size, size))


NOISE_SD_LOW_SNR = 60  # discriminating: ~95% recovered with leave-one-out, ~30% without


def _corr(a, b):
    return np.corrcoef(a.ravel(), b.ravel())[0, 1]


def _aligned_corr(img, truth, margin=6):
    """Correlation with the truth after the best integer translation (interior only)."""
    from skimage.registration import phase_cross_correlation
    shift, _, _ = phase_cross_correlation(truth, img, normalization=None)
    aligned = np.roll(img, tuple(np.round(shift).astype(int)), axis=(-2, -1))
    inner = (slice(margin, -margin), slice(margin, -margin))
    return _corr(aligned[inner], truth[inner])


class LoopVolumeRegistrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "stack.tif"
        self.rng = np.random.default_rng(0)

    def tearDown(self):
        self.tmp.cleanup()

    def test_page_order_without_motion_recovers_each_plane_and_channel(self):
        truth = _truth(self.rng, n_planes=5, n_channels=2, size=48)
        motion = np.zeros((4, 5, 2), dtype=int)
        _write_loop_tiff(self.path, truth, 4, motion, noise_sd=0, rng=self.rng)

        for n_processes in (1, 2):
            stack, shifts = _load_registered_loop_volumes(
                self.path, 5, 4, 2, ref_channel=None,
                block_memory_bytes=1, n_processes=n_processes)  # forces one plane per block
            self.assertEqual(stack.shape, (5, 2, 48, 48))
            np.testing.assert_allclose(stack, np.round(truth), atol=1)
            np.testing.assert_array_equal(shifts, 0)

    def test_volume_motion_is_corrected_before_averaging(self):
        n_planes, n_volumes, size = 3, 24, 64
        truth = _truth(self.rng, n_planes, 2, size)
        motion = self.rng.integers(-3, 4, size=(n_volumes, n_planes, 2))
        _write_loop_tiff(self.path, truth, n_volumes, motion, noise_sd=150, rng=self.rng)

        stack, shifts = _load_registered_loop_volumes(
            self.path, n_planes, n_volumes, 2, ref_channel=0, n_processes=1)

        # Shifts undo the injected motion up to one constant offset per plane (the reference's).
        rel_found = shifts[:, :, 0] - np.median(shifts[:, :, 0], axis=0)
        rel_true = -(motion - np.median(motion, axis=0))
        self.assertGreaterEqual(np.mean(np.abs(rel_found - rel_true) <= 1), 0.95)

        # The registered mean beats the naive mean for every plane (each aligned to the truth,
        # compared away from the edge fill).
        frames = tifffile.imread(self.path).reshape(n_volumes, n_planes, 2, size, size).astype(float)
        naive = frames.mean(axis=0)
        for z in range(n_planes):
            self.assertGreater(_aligned_corr(stack[z, 0], truth[z, 0]), _aligned_corr(naive[z, 0], truth[z, 0]))

    def test_reference_channel_shifts_are_applied_to_other_channel(self):
        truth = _truth(self.rng, 2, 2, 48)
        motion = self.rng.integers(-2, 3, size=(8, 2, 2))
        _write_loop_tiff(self.path, truth, 8, motion, noise_sd=50, rng=self.rng)
        _, shifts = _load_registered_loop_volumes(self.path, 2, 8, 2, ref_channel=1, n_processes=1)
        np.testing.assert_array_equal(shifts[:, :, 0], shifts[:, :, 1])

    def test_low_snr_frames_are_not_biased_to_zero_shift(self):
        # Low SNR, like one frame of a real z-stack: a frame compared against a reference that
        # includes it would lock onto its own noise at zero lag.
        n_volumes, size = 40, 96
        rng = self.rng
        truth = 1000 + 60 * (_truth(rng, 1, 1, size) - 200) / 2000
        motion = rng.choice([-4, -3, 3, 4], size=(n_volumes, 1, 2))
        _write_loop_tiff(self.path, truth, n_volumes, motion, noise_sd=NOISE_SD_LOW_SNR, rng=rng)

        _, shifts = _load_registered_loop_volumes(self.path, 1, n_volumes, 1, n_processes=1)
        rel_found = shifts[:, 0, 0] - np.median(shifts[:, 0, 0], axis=0)
        rel_true = -(motion[:, 0] - np.median(motion[:, 0], axis=0))
        self.assertGreaterEqual(np.mean(np.all(np.abs(rel_found - rel_true) <= 1, axis=1)), 0.9)
        self.assertLess(np.mean(np.all(np.abs(shifts[:, 0, 0]) < 0.05, axis=1)), 0.1)

    def test_summarize_volume_shifts(self):
        shifts = np.zeros((10, 3, 2, 2), dtype=np.float32)
        shifts[3, 1, 0] = [3, 4]  # one 5 px excursion on plane 1, channel 0
        summary = summarize_volume_shifts(shifts)
        self.assertEqual(len(summary), 2)
        self.assertEqual(summary[0]["n_volumes_registered"], 10)
        self.assertEqual(summary[0]["displacement_px_max"], 5.0)
        self.assertEqual(summary[1]["displacement_px_max"], 0.0)


class BetweenPlaneGuardTest(unittest.TestCase):
    def test_failed_plane_keeps_neighbour_shift_and_is_flagged(self):
        rng = np.random.default_rng(1)
        base = _truth(rng, 1, 1, 96)[0, 0]
        stack = np.stack([np.roll(base, (z // 4, 0), axis=(0, 1)) for z in range(20)]).astype(np.float32)
        stack[14] = ndi.shift(stack[14], (60, -45), mode="wrap")  # a plane that registers far away

        _, shifts, flagged = reg_between_planes(stack, ref_ind=5, return_flagged=True)
        self.assertIn(14, flagged)
        np.testing.assert_allclose(shifts[14], shifts[13])
        self.assertTrue(all(np.linalg.norm(np.diff(np.array(shifts), axis=0), axis=1) <= 10 + 1e-6))

        _, unguarded = reg_between_planes(stack, ref_ind=5, max_step_px=None)
        self.assertGreater(np.linalg.norm(unguarded[14] - unguarded[13]), 10)


if __name__ == "__main__":
    unittest.main()
