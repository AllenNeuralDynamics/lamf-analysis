import unittest

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from lamf_analysis.behavior.engagement import (
    consumed_auto_reward_mask,
    engagement_state,
    lick_bout_rate,
    reward_rate,
)
from lamf_analysis.behavior.licks import get_lick_bout_start_times
from lamf_analysis.ophys.stimulus_processing import (
    annotate_flash_rolling_metrics,
)


class EngagementTest(unittest.TestCase):
    def test_get_lick_bout_start_times_groups_close_licks(self):
        lick_times = [10.0, 10.2, 10.5, 11.01, 20.0]

        result = get_lick_bout_start_times(lick_times)

        np.testing.assert_allclose(result, [10.0, 20.0])

    def test_lick_bout_rate_is_gaussian_rate_in_bouts_per_minute(self):
        sample_times = np.arange(0.0, 61.0)
        lick_times = np.array([30.0, 30.2])

        result = lick_bout_rate(
            sample_times,
            lick_times,
            gaussian_sd_s=1.0,
        )

        self.assertAlmostEqual(
            result[30],
            60.0 * gaussian_filter1d(
                np.eye(1, 61, 30)[0],
                sigma=1.0,
                mode="nearest",
            )[30],
        )

    def test_reward_rate_uses_centered_edge_clipped_window(self):
        result = reward_rate(
            sample_times=[0.0, 100.0, 200.0],
            reward_times=[100.0],
            window_s=200.0,
        )

        np.testing.assert_allclose(result, [1 / 100, 1 / 200, 1 / 100])

    def test_consumed_auto_reward_mask_requires_post_reward_lick(self):
        result = consumed_auto_reward_mask(
            sample_times=[0, 1, 5, 10, 12, 15],
            lick_times=[3, 20],
            auto_reward_times=[0, 10],
            post_reward_window_s=5,
        )

        np.testing.assert_array_equal(
            result,
            [True, True, True, False, False, False],
        )

    def test_lick_engagement_labels_consumed_auto_reward_period(self):
        _, state = engagement_state(
            sample_times=[0, 1, 2, 11],
            method="lick",
            lick_times=[1],
            auto_reward_times=[0],
            lick_rate_threshold_per_min=0,
        )

        np.testing.assert_array_equal(
            state,
            ["autoreward", "autoreward", "autoreward", "disengaged"],
        )

    def test_annotate_reward_preserves_flash_onset_reward_convention(self):
        sp_df = pd.DataFrame(
            {
                "start_time": [0.0, 100.0, 200.0],
                "rewards": [[], [100.2], []],
            }
        )

        result = annotate_flash_rolling_metrics(
            sp_df,
            200.0,
            reward_rate_threshold_per_s=0.006,
            engagement_method="reward",
        )

        np.testing.assert_allclose(
            result["reward_rate"],
            [0.01, 0.005, 0.01],
        )
        self.assertEqual(
            result["engagement_state"].tolist(),
            ["engaged", "disengaged", "engaged"],
        )

    def test_annotate_lick_uses_raw_event_inputs(self):
        sp_df = pd.DataFrame({"start_time": [0.0, 1.0, 11.0]})

        result = annotate_flash_rolling_metrics(
            sp_df,
            licks=np.array([1.0]),
            auto_rewards=np.array([0.0]),
            lick_gaussian_sd_s=1.0,
            lick_rate_threshold_per_min=0,
        )

        self.assertIn("lick_bout_rate", result)
        self.assertEqual(
            result["engagement_state"].tolist(),
            ["autoreward", "autoreward", "disengaged"],
        )

    def test_annotate_rejects_unknown_engagement_method(self):
        sp_df = pd.DataFrame({"start_time": [0.0]})

        with self.assertRaisesRegex(
            ValueError,
            "engagement_method must be 'reward' or 'lick'",
        ):
            annotate_flash_rolling_metrics(
                sp_df,
                engagement_method="running",
            )

    def test_legacy_behavior_utils_imports_are_available(self):
        from lamf_analysis.behavior import utils

        self.assertIsNotNone(utils.add_lick_bouts)
        self.assertIsNotNone(utils.get_running_epochs)


if __name__ == "__main__":
    unittest.main()
