import unittest

import numpy as np
import pandas as pd
from scipy.stats import norm

from lamf_analysis.behavior.task_performance import (
    matched_flash_false_alarm_rate,
    per_flash_false_alarm_rate,
    per_time_false_alarm_rate,
    signal_detection_metrics,
    task_performance_metrics,
    trial_performance_metrics,
)


def event(name, time, frame, subtype=""):
    return (name, subtype, time, frame)


class TaskPerformanceTest(unittest.TestCase):
    def test_signal_detection_metrics_uses_loglinear_limits(self):
        dprime, criterion = signal_detection_metrics(1.0, 0.0, 10, 20)

        hit_z = norm.ppf(0.95)
        false_alarm_z = norm.ppf(0.025)
        self.assertAlmostEqual(dprime, hit_z - false_alarm_z)
        self.assertAlmostEqual(criterion, -0.5 * (hit_z + false_alarm_z))

    def test_trial_metrics_exclude_autoreward_and_catch_from_abort_rate(self):
        trials = [
            {
                "trial_params": {},
                "events": [
                    event("pre_change", 1, 1, "enter"),
                    event("stimulus_changed", 4, 4),
                    event("hit", 4.2, 4),
                ],
                "rewards": [(0.005,)],
            },
            {
                "trial_params": {},
                "events": [
                    event("pre_change", 10, 10, "enter"),
                    event("stimulus_changed", 14, 14),
                    event("miss", 14.2, 14),
                ],
                "rewards": [],
            },
            {
                "trial_params": {},
                "events": [
                    event("pre_change", 20, 20, "enter"),
                    event("abort", 22, 22),
                ],
                "rewards": [],
            },
            {
                "trial_params": {"auto_reward": True},
                "events": [event("stimulus_changed", 30, 30)],
                "rewards": [(0.005,)],
            },
            {
                "trial_params": {"catch": True},
                "events": [
                    event("sham_change", 40, 40),
                    event("false_alarm", 40.2, 40),
                ],
                "rewards": [],
            },
            {
                "trial_params": {"catch": True},
                "events": [
                    event("sham_change", 50, 50),
                    event("rejection", 50.2, 50),
                ],
                "rewards": [],
            },
        ]

        result = trial_performance_metrics(
            trials,
            stimulus_timestamps=np.arange(60, dtype=float),
            engagement_times=np.arange(60, dtype=float),
            engaged=np.ones(60, dtype=bool),
            monitor_delay_s=0,
        )

        self.assertEqual(result["n_trials"], 6)
        self.assertEqual(result["n_go"], 4)
        self.assertEqual(result["n_catch"], 2)
        self.assertEqual(result["n_autoreward"], 1)
        self.assertEqual(result["hit"], 1)
        self.assertEqual(result["miss"], 1)
        self.assertEqual(result["false_alarm"], 1)
        self.assertEqual(result["correct_reject"], 1)
        self.assertEqual(result["abort_false_alarm_rate"], 1 / 3)
        self.assertEqual(result["premature_rate_per_s"], 1 / 9)
        self.assertEqual(result["reward_ul"], 10.0)
        self.assertEqual(result["hit_rate_engaged"], 0.5)
        self.assertEqual(result["catch_false_alarm_rate_engaged"], 0.5)
        self.assertEqual(result["session_duration_min"], 59 / 60)
        self.assertEqual(result["engaged_time_min"], 1.0)

    def test_monitor_delay_aligns_event_frame_to_engagement_time(self):
        trials = [
            {
                "trial_params": {},
                "events": [
                    event("pre_change", 0, 0, "enter"),
                    event("stimulus_changed", 1, 1),
                    event("hit", 1.2, 1),
                ],
                "rewards": [],
            }
        ]

        without_delay = trial_performance_metrics(
            trials,
            stimulus_timestamps=[0.0, 1.0],
            engagement_times=[0.0, 1.0, 1.02, 2.0],
            engaged=[False, False, True, True],
            monitor_delay_s=0,
        )
        with_delay = trial_performance_metrics(
            trials,
            stimulus_timestamps=[0.0, 1.0],
            engagement_times=[0.0, 1.0, 1.02, 2.0],
            engaged=[False, False, True, True],
            monitor_delay_s=0.03613,
        )

        self.assertEqual(without_delay["n_go_engaged"], 0)
        self.assertEqual(with_delay["n_go_engaged"], 1)

    def test_per_flash_false_alarm_uses_only_clean_engaged_flashes(self):
        presentations = pd.DataFrame(
            {
                "start_time": np.arange(8, dtype=float),
                "omitted": [False, False, False, False, False, True, False, False],
                "is_change": [True, False, False, False, False, False, False, False],
                "flashes_since_change": [0, 1, 2, 3, 4, 4, 4, 5],
            }
        )

        result = per_flash_false_alarm_rate(
            presentations,
            lick_times=[4.2, 7.2],
            engagement_times=np.arange(8, dtype=float),
            engaged=[True, True, True, True, True, True, True, False],
            min_clean_presentations=1,
            min_session_presentations=0,
        )

        self.assertEqual(result["n_clean"], 2)
        self.assertEqual(result["n_clean_engaged"], 1)
        self.assertEqual(result["false_alarm_rate"], 1.0)
        self.assertEqual(result["false_alarm_rate_engaged"], 1.0)

    def test_matched_flash_false_alarm_weights_change_positions(self):
        presentations = pd.DataFrame(
            {
                "start_time": np.arange(12, dtype=float),
                "omitted": np.zeros(12, dtype=bool),
                "is_change": [
                    True,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                ],
                "flashes_since_change": [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 0],
            }
        )

        result = matched_flash_false_alarm_rate(
            presentations,
            lick_times=[10.2],
            engagement_times=np.arange(12, dtype=float),
            engaged=np.ones(12, dtype=bool),
            min_clean_presentations=1,
            min_session_presentations=0,
        )

        self.assertEqual(result["false_alarm_rate"], 1.0)
        self.assertEqual(result["false_alarm_rate_engaged"], 1.0)

    def test_per_time_false_alarm_is_undefined_when_no_licks(self):
        trials = [
            {
                "events": [event("stimulus_changed", time, frame)],
                "trial_params": {},
            }
            for time, frame in [(5, 5), (10, 10), (15, 15), (20, 20), (25, 25)]
        ]

        result = per_time_false_alarm_rate(
            trials,
            stimulus_timestamps=np.arange(31, dtype=float),
            lick_times=[],
            engagement_times=np.arange(31, dtype=float),
            engaged=np.ones(31, dtype=bool),
            monitor_delay_s=0,
            min_samples=1,
        )

        self.assertEqual(result["n_samples"], 0)
        self.assertTrue(np.isnan(result["false_alarm_rate"]))
        self.assertTrue(np.isnan(result["false_alarm_rate_engaged"]))

    def test_combined_metrics_derives_all_false_alarm_dprime_variants(self):
        n_presentations = 1001
        start_times = np.arange(n_presentations, dtype=float) * 0.75
        change_indices = np.array([100, 300, 500, 700, 900])
        is_change = np.zeros(n_presentations, dtype=bool)
        is_change[change_indices] = True
        flashes_since_change = np.zeros(n_presentations, dtype=int)
        since_change = 0
        for index in range(n_presentations):
            if is_change[index]:
                since_change = 0
            else:
                since_change += 1
            flashes_since_change[index] = since_change
        presentations = pd.DataFrame(
            {
                "start_time": start_times,
                "omitted": np.zeros(n_presentations, dtype=bool),
                "is_change": is_change,
                "flashes_since_change": flashes_since_change,
            }
        )
        trials = [
            {
                "trial_params": {},
                "events": [
                    event("pre_change", frame - 5, frame - 5, "enter"),
                    event("stimulus_changed", frame, frame),
                    event("hit", frame + 0.2, frame),
                ],
                "rewards": [],
            }
            for frame in change_indices
        ]
        lick_times = np.r_[
            start_times[50::100] + 0.2,
            change_indices.astype(float) + 0.2,
        ]
        engagement_times = np.arange(n_presentations, dtype=float)

        result = task_performance_metrics(
            trials,
            stimulus_timestamps=np.arange(n_presentations, dtype=float),
            stimulus_presentations=presentations,
            lick_times=lick_times,
            engagement_times=engagement_times,
            engaged=np.ones(n_presentations, dtype=bool),
            monitor_delay_s=0,
        )

        for suffix in (
            "per_flash",
            "per_flash_engaged",
            "per_time",
            "per_time_engaged",
        ):
            self.assertIn(f"dprime_{suffix}", result)
            self.assertIn(f"criterion_{suffix}", result)
            self.assertTrue(np.isfinite(result[f"dprime_{suffix}"]))
            self.assertTrue(np.isfinite(result[f"criterion_{suffix}"]))

    def test_rejects_mismatched_engagement_inputs(self):
        with self.assertRaisesRegex(ValueError, "equal length"):
            trial_performance_metrics(
                [],
                stimulus_timestamps=[0],
                engagement_times=[0, 1],
                engaged=[True],
            )

    def test_rejects_string_engagement_states(self):
        with self.assertRaisesRegex(ValueError, "boolean"):
            trial_performance_metrics(
                [],
                stimulus_timestamps=[0],
                engagement_times=[0],
                engaged=["disengaged"],
            )

    def test_per_time_rejects_negative_monitor_delay(self):
        with self.assertRaisesRegex(ValueError, "monitor_delay_s"):
            per_time_false_alarm_rate(
                [],
                stimulus_timestamps=[0],
                lick_times=[],
                engagement_times=[0],
                engaged=[True],
                monitor_delay_s=-0.1,
            )


if __name__ == "__main__":
    unittest.main()
