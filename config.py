from dataclasses import dataclass


@dataclass(frozen=True)
class GestureThresholds:
    pinch_distance: float = 0.05
    pinch_press_threshold: float = 0.45
    pinch_release_threshold: float = 0.60
    click_max_duration: float = 0.350
    pinch_freeze_frames: int = 3
    pinch_min_cutoff: float = 2.0
    pinch_beta: float = 0.02
    drag_hold_ms: float = 400.0
    min_pinch_ms: float = 40.0
    click_flash_ms: float = 400.0
    right_press_threshold: float = 0.48
    right_release_threshold: float = 0.63
    right_click_max_duration: float = 0.30
    finger_extension_threshold: float = 1.15
    scroll_remainder: bool = True
    scroll_sensitivity: float = 1200.0
    scroll_dead_zone: float = 0.002
    scroll_invert: bool = False
    scroll_min_cutoff: float = 1.0
    scroll_beta: float = 0.007
    pointer_anchor: str = "palm"
    palm_anchor_mcp_weight: float = 0.70
    pointer_mode: str = "absolute"
    pointer_gain: float = 1.0
    pointer_acceleration: float = 0.35
    pointer_speed_reference: float = 0.8
    invert_x: bool = False
    invert_y: bool = False
    output_rate_hz: float = 120.0
    output_smoothing_time_constant: float = 0.08
    stillness_threshold: float = 0.004
    pointer_history_size: int = 32
    latch_ms: float = 120.0
    calibration_file: str = "calibration.json"
    calibration_countdown_seconds: float = 3.0
    calibration_sample_seconds: float = 1.0
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30
    extended_finger_margin: float = 0.02
    active_region_margin: float = 0.20
    one_euro_min_cutoff: float = 1.0
    one_euro_beta: float = 0.007
    pose_stable_frames: int = 5
    pose_exit_frames: int = 3
    double_click_ms: float = 400.0
    clutch_hold_ms: float = 700.0


GESTURE_THRESHOLDS = GestureThresholds()
