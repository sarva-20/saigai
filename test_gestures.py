import unittest

from gestures import (
    OneEuroFilter,
    PinchStateMachine,
    map_index_finger_to_screen,
    normalized_pinch_distance,
)


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

    def test_normalized_pinch_distance_ignores_hand_scale(self) -> None:
        small = [[0.0, 0.0, 0.0] for _ in range(21)]
        large = [[0.0, 0.0, 0.0] for _ in range(21)]
        small[9] = [0.1, 0.0, 0.0]
        small[4] = [0.0, 0.02, 0.0]
        small[8] = [0.0, 0.04, 0.0]
        large[9] = [0.5, 0.0, 0.0]
        large[4] = [0.0, 0.1, 0.0]
        large[8] = [0.0, 0.2, 0.0]

        self.assertAlmostEqual(
            normalized_pinch_distance(small),
            normalized_pinch_distance(large),
        )

    def test_pinch_state_machine_hysteresis_click_and_drag(self) -> None:
        state = PinchStateMachine(press_threshold=0.45, release_threshold=0.60)

        self.assertTrue(state.update(0.40, 0.0).pressed)
        self.assertTrue(state.update(0.50, 0.1).dragging)
        self.assertFalse(state.update(0.50, 0.2).released)
        click_release = state.update(0.61, 0.2)
        self.assertTrue(click_release.released)
        self.assertTrue(click_release.clicked)

        self.assertTrue(state.update(0.40, 1.0).pressed)
        self.assertTrue(state.update(0.40, 1.4).dragging)
        drag_release = state.update(0.61, 1.5)
        self.assertTrue(drag_release.released)
        self.assertFalse(drag_release.clicked)

        self.assertTrue(state.update(0.40, 2.0).pressed)
        lost_hand_release = state.update(float("inf"), 2.1)
        self.assertTrue(lost_hand_release.released)


if __name__ == "__main__":
    unittest.main()
