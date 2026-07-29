"""Tuned constants for the torque v2.1 output governor.

These live in a private module so the pure-Python reference and the Cython helper
module share the same values without public API changes.
"""

# --- RATE-LIMIT: slew schedules (refined governor) ---
# Routes 00000296/297: Toyota's downstream 15-up/25-down raw limits bound ~20% of
# hands-off frames and produce the notchy wheel-angle/jerk tail. Keep v2.1 just below
# those limits: 0.8/s = 12 raw/frame build and 1.25/s = 18.75 raw/frame release at
# STEER_MAX=1500. Driver and safety releases still bypass this comfort backstop.
OUTPUT_SLEW_RATE_BP = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0]
OUTPUT_SLEW_RATE_V = [0.80, 0.80, 0.80, 0.80, 0.80, 0.80]
SIGN_CHANGE_SLEW_RATE_BP = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0]
SIGN_CHANGE_SLEW_RATE_V = [1.25, 1.25, 1.25, 1.25, 1.25, 1.25]
SAME_DIRECTION_LIMIT_RATE_BP = [0.0, 10.0, 20.0, 30.0, 40.0]
SAME_DIRECTION_LIMIT_RATE_V = [1.30, 1.30, 2.10, 3.20, 3.60]
# Same-sign decreases used to bypass the slew entirely, so any cap engagement or demand
# collapse stepped torque down in one frame (IMU survey catch-down snaps). 1.5625x the build
# slew keeps faithful unwind untouched and only spreads step discontinuities over a few
# frames; driver release and safety cuts (sign conflict / over-response / ISO) still drop
# instantly.
RELEASE_SLEW_SCALE = 1.5625
# Slew-scale study (LateralSlewScaleMode): apply multiplies the BUILD slew only by
# 1.125x — 0.90/s = 13.5 raw/frame (< Toyota's 15 up). Sign-change/release scaling was
# rejected 2026-07-20 (routes 2ba vs 2bb/2bc): the release ceiling binds at the applied
# torque-rate p95, so scaling it sharpened catch-down steps ("twitchy") and partially
# undid the RELEASE_SLEW_SCALE smoothing above.
SLEW_RATE_SCALE_STEP = 1.125

# Manual steering reaches peak wheel rate near mid-stroke, then blends into holding
# torque. Begin the blend one actuator delay plus 0.25 s before predicted arrival and
# finish with 0.05 s remaining. The closing-rate floor rejects noisy near-static timing.
TARGET_ARRIVAL_TAPER_START = 0.25
TARGET_ARRIVAL_TAPER_FULL = 0.05
TARGET_ARRIVAL_MIN_CLOSING_RATE = 0.05

# --- RESTRICT: caps ---
SAME_DIRECTION_LIMIT_CAP = 0.85
STEERING_RATE_COMFORT_START_DEG = 25.0
STEERING_RATE_COMFORT_FULL_DEG = 80.0
STEERING_RATE_COMFORT_MIN_CAP = 0.88
STEERING_RATE_COMFORT_MIN_SLEW_SCALE = 0.75
HIGH_RATE_START_DEG = 80.0
HIGH_RATE_FULL_DEG = 100.0
HIGH_RATE_MIN_CAP = 0.62
HIGH_RATE_SLEW_SCALE = 0.70
SIGN_CONFLICT_CAP = 0.80
OVERRIDE_RELEASE_CAP = 0.80
OVER_RESPONSE_MARGIN = 0.08
OVER_RESPONSE_FULL_EXCESS = 0.45
OVER_RESPONSE_MIN_SCALE = 0.30
ISO_LATERAL_ACCEL = 3.0
ISO_ACCEL_MARGIN = 2.6
NEAR_ISO_ACCEL_CAP = 0.85
OVER_ISO_ACCEL_CAP = 0.80
UNDER_RESPONSE_MAX_TORQUE_FRACTION = 0.90
TRACKING_CORRECTION_MARGIN = 0.06

# nuPlan comfort bounds (reference thresholds; not currently active limits).
NUPLAN_COMFORT_LAT_ACCEL = 4.89
NUPLAN_COMFORT_JERK = 8.37

# --- AUGMENT: under-response floor ---
UNDER_RESPONSE_MARGIN = 0.12
UNDER_RESPONSE_FULL_SPEED = 9.0
UNDER_RESPONSE_FADE_SPEED = 12.0

SIGN_THRESHOLD = 0.05
