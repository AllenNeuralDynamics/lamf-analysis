import unittest

import pandas as pd

from lamf_analysis.ophys.motion_border_utils import get_max_correction_from_df


class MotionBorderUtilsTest(unittest.TestCase):
    def test_max_correction_excludes_outliers_and_preserves_directions(self):
        shifts = pd.DataFrame({
            "x": [-100.0, -3.0, 2.0, 5.0, 100.0],
            "y": [-100.0, -4.0, 1.0, 6.0, 100.0],
        })

        result = get_max_correction_from_df(shifts, max_shift=10.0)

        self.assertEqual(result.left, 5.0)
        self.assertEqual(result.right, 3.0)
        self.assertEqual(result.up, 6.0)
        self.assertEqual(result.down, 4.0)


if __name__ == "__main__":
    unittest.main()
