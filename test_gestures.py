import unittest

from gestures import (
    HandInteractionStateMachine,
    ModeStateMachine,
    PoseDebouncer,
    OneEuroFilter,
    PointerHistory,
    RatioMedianFilter,
    PinchStateMachine,
    ScrollStateMachine,
    StillnessLock,
    get_pointer_anchor,
    is_scroll_pose,
    map_index_finger_to_screen,
    normalized_pinch_distance,
    normalized_thumb_middle_distance,
    relative_pointer_delta,
    calibrated_pinch_thresholds,
    classify_hand_pose,
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

        self.assertEqual(x, 0.0)
        self.assertEqual(y, 0.0)

        right_up = map_index_finger_to_screen(
            (0.8, 0.2, 0.0),
            100,
            100,
            1000,
            500,
            0.20,
        )
        self.assertGreater(right_up[0], x)
        self.assertEqual(right_up[1], y)

    def test_pointer_anchor_defaults_to_palm_and_supports_fingertip(self) -> None:
        landmarks = [[0.0, 0.0, 0.0] for _ in range(21)]
        landmarks[0] = [0.0, 0.0, 0.0]
        landmarks[5] = [1.0, 0.5, 0.0]
        landmarks[8] = [0.2, 0.8, 0.0]

        self.assertEqual(get_pointer_anchor(landmarks), (0.7, 0.35, 0.0))
        self.assertEqual(
            get_pointer_anchor(landmarks, "fingertip"),
            (0.2, 0.8, 0.0),
        )

    def test_relative_pointer_delta_applies_gain_and_acceleration(self) -> None:
        delta = relative_pointer_delta(
            (0.5, 0.5, 0.0),
            (0.6, 0.5, 0.0),
            delta_time=1.0,
            screen_width=1000,
            screen_height=1000,
            gain=1.0,
            acceleration=0.0,
        )
        self.assertAlmostEqual(delta[0], 100.0)
        self.assertAlmostEqual(delta[1], 0.0)

    def test_inversion_escape_hatches(self) -> None:
        normal = map_index_finger_to_screen(
            (0.8, 0.2, 0.0), 100, 100, 1000, 500, 0.20
        )
        inverted = map_index_finger_to_screen(
            (0.8, 0.2, 0.0),
            100,
            100,
            1000,
            500,
            0.20,
            invert_x=True,
            invert_y=True,
        )
        self.assertEqual(normal, (1000.0, 0.0))
        self.assertEqual(inverted, (0.0, 500.0))

    def test_stillness_lock_releases_after_motion(self) -> None:
        lock = StillnessLock(0.01)
        self.assertFalse(lock.update((0.5, 0.5, 0.0)))
        self.assertTrue(lock.update((0.505, 0.505, 0.0)))
        self.assertFalse(lock.update((0.6, 0.5, 0.0)))

    def test_pointer_history_latches_earlier_position(self) -> None:
        history = PointerHistory(4)
        history.add((10.0, 10.0), 1.00)
        history.add((20.0, 20.0), 1.05)
        history.add((30.0, 30.0), 1.10)
        self.assertEqual(history.latch(1.20, 160), (10.0, 10.0))
        self.assertEqual(history.latch(1.20, 110), (20.0, 20.0))

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

    def test_normalized_thumb_middle_distance_is_handedness_independent(self) -> None:
        landmarks = [[0.0, 0.0, 0.0] for _ in range(21)]
        landmarks[9] = [0.2, 0.0, 0.0]
        landmarks[4] = [0.0, 0.04, 0.0]
        landmarks[12] = [0.0, 0.08, 0.0]
        self.assertAlmostEqual(normalized_thumb_middle_distance(landmarks), 0.20)

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

    @staticmethod
    def scroll_hand() -> list[list[float]]:
        landmarks = [[0.0, 0.0, 0.0] for _ in range(21)]
        landmarks[6] = [0.15, 0.15, 0.0]
        landmarks[8] = [0.30, 0.30, 0.0]
        landmarks[10] = [-0.15, 0.15, 0.0]
        landmarks[12] = [-0.30, 0.30, 0.0]
        landmarks[14] = [-0.15, -0.15, 0.0]
        landmarks[16] = [-0.10, -0.10, 0.0]
        landmarks[18] = [0.15, -0.15, 0.0]
        landmarks[20] = [0.10, -0.10, 0.0]
        return landmarks

    def test_scroll_pose_requires_extended_index_middle_and_curled_others(self) -> None:
        landmarks = self.scroll_hand()
        self.assertTrue(is_scroll_pose(landmarks))
        self.assertFalse(is_scroll_pose(landmarks, pinch_active=True))
        landmarks[16] = [0.30, 0.30, 0.0]
        self.assertFalse(is_scroll_pose(landmarks))

    def test_scroll_pose_at_tilted_hand_angle(self) -> None:
        landmarks = self.scroll_hand()
        tilted = [[point[0] - 0.10, point[1] + 0.25, point[2]] for point in landmarks]
        for index in (8, 12):
            tilted[index][0] += 0.20
            tilted[index][1] += 0.20
        self.assertTrue(is_scroll_pose(tilted))

    def test_scroll_dead_zone_and_direction_inversion(self) -> None:
        landmarks = self.scroll_hand()
        state = ScrollStateMachine(100.0, dead_zone=0.01)
        self.assertTrue(state.update(landmarks, 0.0).started)
        self.assertEqual(state.update(landmarks, 1.0).delta, 0.0)
        landmarks[8][1] += 0.1
        landmarks[12][1] += 0.1
        natural = state.update(landmarks, 2.0).delta
        self.assertLess(natural, 0.0)

        inverted = ScrollStateMachine(100.0, dead_zone=0.01, invert=True)
        inverted.update(self.scroll_hand(), 0.0)
        inverted_landmarks = self.scroll_hand()
        inverted_landmarks[8][1] += 0.1
        inverted_landmarks[12][1] += 0.1
        self.assertGreater(inverted.update(inverted_landmarks, 2.0).delta, 0.0)

    def test_pinch_disables_scroll_and_scroll_ends_before_cursor_mode(self) -> None:
        landmarks = self.scroll_hand()
        state = ScrollStateMachine(100.0, dead_zone=0.0)
        self.assertTrue(state.update(landmarks, 0.0).active)
        self.assertTrue(state.update(landmarks, 1.0, pinch_active=True).ended)
        self.assertFalse(state.update(landmarks, 2.0, pinch_active=True).active)

        pinch = PinchStateMachine(0.45, 0.60)
        pinch_press = pinch.update(0.40, 3.0)
        self.assertTrue(pinch_press.pressed)
        self.assertFalse(
            state.update(landmarks, 3.0, pinch_active=pinch.pressed).active
        )

    def test_interaction_states_click_drag_and_right_click_exclusivity(self) -> None:
        state = HandInteractionStateMachine(
            0.45,
            0.60,
            0.48,
            0.63,
            drag_hold_ms=250,
            click_max_duration=0.30,
            click_flash_ms=400,
        )
        pressed = state.update(0.40, 0.70, 0.0)
        self.assertEqual(pressed.state, "PINCHING")
        dragging = state.update(0.40, 0.70, 0.26)
        self.assertEqual(dragging.state, "PINCHING")
        click_end = state.update(0.61, 0.70, 0.299)
        self.assertTrue(click_end.left_clicked)

        right = HandInteractionStateMachine(
            0.45, 0.60, 0.48, 0.63, drag_hold_ms=250
        )
        self.assertEqual(right.update(0.70, 0.40, 1.0).state, "PINCHING")
        right_click = right.update(0.70, 0.64, 1.1)
        self.assertTrue(right_click.right_clicked)

        exclusive = HandInteractionStateMachine(
            0.45, 0.60, 0.48, 0.63
        )
        event = exclusive.update(0.40, 0.40, 2.0)
        self.assertEqual(event.state, "PINCHING")
        self.assertTrue(exclusive.mode == "left")

    def test_scroll_remainder_carries_fractional_amount(self) -> None:
        landmarks = self.scroll_hand()
        state = ScrollStateMachine(100.0, dead_zone=0.0)
        state.update(landmarks, 0.0)
        landmarks[8][1] += 0.002
        landmarks[12][1] += 0.002
        first = state.update(landmarks, 1.0)
        self.assertLessEqual(abs(first.sent_amount), 1)
        landmarks[8][1] += 0.002
        landmarks[12][1] += 0.002
        second = state.update(landmarks, 2.0)
        combined = first.sent_amount + second.sent_amount
        self.assertGreaterEqual(combined, 0)

    def test_calibration_threshold_math(self) -> None:
        press, release = calibrated_pinch_thresholds(1.0, 0.2)
        self.assertAlmostEqual(press, 0.48)
        self.assertAlmostEqual(release, 0.64)

    def test_median_filter_uses_last_three_ratios(self) -> None:
        median_filter = RatioMedianFilter(3)
        self.assertEqual(median_filter.update(0.8), 0.8)
        self.assertEqual(median_filter.update(0.2), 0.5)
        self.assertEqual(median_filter.update(0.4), 0.4)
        self.assertEqual(median_filter.update(0.1), 0.2)

    def test_minimum_pinch_duration_guard(self) -> None:
        state = HandInteractionStateMachine(
            0.45, 0.60, 0.48, 0.63,
            drag_hold_ms=400,
            click_max_duration=0.350,
            min_pinch_ms=40,
        )
        self.assertTrue(state.update(0.40, 0.70, 1.0).left_pressed)
        ignored = state.update(0.61, 0.70, 1.039)
        self.assertEqual(ignored.state, "READY")
        self.assertFalse(ignored.left_clicked)

    def test_click_and_drag_boundaries(self) -> None:
        click_state = HandInteractionStateMachine(
            0.45, 0.60, 0.48, 0.63,
            drag_hold_ms=400,
            click_max_duration=0.350,
            min_pinch_ms=40,
        )
        click_state.update(0.40, 0.70, 0.0)
        under = click_state.update(0.61, 0.70, 0.349)
        self.assertTrue(under.left_clicked)

        at_drag = HandInteractionStateMachine(
            0.45, 0.60, 0.48, 0.63,
            drag_hold_ms=400,
            click_max_duration=0.350,
            min_pinch_ms=40,
        )
        at_drag.update(0.40, 0.70, 0.0)
        at_boundary = at_drag.update(0.40, 0.70, 0.400)
        self.assertEqual(at_boundary.state, "PINCHING")

    @staticmethod
    def pose_hand(count: int) -> list[list[float]]:
        landmarks = [[0.0, 0.0, 0.0] for _ in range(21)]
        wrists = (8, 12, 16, 20)
        pips = (6, 10, 14, 18)
        for finger, (tip, pip) in enumerate(zip(wrists, pips)):
            landmarks[pip] = [0.2, 0.0, 0.0]
            landmarks[tip] = [0.1, 0.0, 0.0]
        for finger, (tip, pip) in enumerate(zip(wrists, pips)):
            if finger < count:
                landmarks[pip] = [0.1, 0.0, 0.0]
                landmarks[tip] = [0.3, 0.0, 0.0]
        landmarks[0] = [0.0, 0.0, 0.0]
        return landmarks

    def test_pose_classification_all_modes_and_tilt(self) -> None:
        self.assertEqual(classify_hand_pose(self.pose_hand(1)), "POINT")
        self.assertEqual(classify_hand_pose(self.pose_hand(2)), "SCROLL")
        self.assertEqual(classify_hand_pose(self.pose_hand(3)), "THREE")
        self.assertEqual(classify_hand_pose(self.pose_hand(4)), "FOUR")
        open_hand = self.pose_hand(4)
        open_hand[20] = [0.3, 0.0, 0.0]
        open_hand[18] = [0.1, 0.0, 0.0]
        open_hand[4] = [0.3, 0.0, 0.0]
        open_hand[3] = [0.1, 0.0, 0.0]
        self.assertEqual(classify_hand_pose(open_hand), "OPEN")
        fist = self.pose_hand(0)
        for tip in (8, 12, 16, 20):
            fist[tip] = [0.05, 0.0, 0.0]
        self.assertEqual(classify_hand_pose(fist), "FIST")

    def test_pose_debounce_and_exit_hysteresis(self) -> None:
        debounce = PoseDebouncer(stable_frames=5, exit_frames=3)
        active = None
        for _ in range(4):
            active = debounce.update("POINT")
        self.assertIsNone(active)
        active = debounce.update("POINT")
        self.assertEqual(active, "POINT")
        self.assertEqual(debounce.update("SCROLL"), "POINT")
        self.assertEqual(debounce.update("SCROLL"), "POINT")
        self.assertIsNone(debounce.update("SCROLL"))

    def test_three_finger_drag_lifecycle_and_hand_loss(self) -> None:
        state = ModeStateMachine(clutch_hold_ms=700)
        entered = state.update_pose("THREE", 1.0)
        self.assertTrue(entered.drag_started)
        self.assertEqual(entered.mode, "THREE")
        ended = state.update_pose("POINT", 2.0)
        self.assertTrue(ended.drag_ended)
        state.update_pose("THREE", 3.0)
        lost = state.hand_lost()
        self.assertTrue(lost.drag_ended)
        self.assertEqual(lost.pose, "NO HAND")

    def test_mode_exclusivity_and_double_click(self) -> None:
        state = ModeStateMachine(double_click_ms=400)
        state.update_pose("POINT", 0.0)
        self.assertEqual(state.mode, "POINT")
        self.assertEqual(state.left_click(1.0), 1)
        self.assertEqual(state.left_click(1.3), 2)
        state.update_pose("SCROLL", 2.0)
        self.assertEqual(state.mode, "SCROLL")

    def test_clutch_toggle_timing(self) -> None:
        state = ModeStateMachine(clutch_hold_ms=700)
        state.update_pose("FIST", 0.0)
        state.update_pose("FIST", 0.699)
        self.assertFalse(state.paused)
        toggled = state.update_pose("FIST", 0.700)
        self.assertTrue(toggled.clutch_toggled)
        self.assertTrue(state.paused)
        state.update_pose("POINT", 1.0)
        resumed = state.update_pose("FIST", 1.1)
        resumed = state.update_pose("FIST", 1.800)
        self.assertTrue(resumed.clutch_toggled)
        self.assertFalse(state.paused)


if __name__ == "__main__":
    unittest.main()
