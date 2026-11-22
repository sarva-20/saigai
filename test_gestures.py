import unittest

from gestures import OneEuroFilter, map_index_finger_to_screen


class GestureTests(unittest.TestCase):
    def test_active_region_mapping_and_mirror(self) -> None:
        x, y = map_index_finger_to_screen(
            (0.2, 0.2, 0.0),
            frame_width=100,
            frame_height=100,
            screen_width=1000,
            screen_height=500,
            active_region_margin=0.20,
        )

        self.assertEqual(x, 1000.0)
        self.assertEqual(y, 0.0)

    def test_one_euro_filter_smooths_movement(self) -> None:
        smoothing = OneEuroFilter(min_cutoff=1.0, beta=0.007)

        first = smoothing(0.0, 0.0)
        second = smoothing(100.0, 1.0)
        third = smoothing(100.0, 2.0)

        self.assertEqual(first, 0.0)
        self.assertGreater(second, 0.0)
        self.assertLess(second, 100.0)
        self.assertGreater(third, second)
        self.assertLess(third, 100.0)


if __name__ == "__main__":
    unittest.main()
