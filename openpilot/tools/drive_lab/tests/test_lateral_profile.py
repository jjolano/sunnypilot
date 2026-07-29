"""Tests for lateral route profiling."""
from __future__ import annotations

from openpilot.tools.drive_lab.log_profile import (
    LateralProfile,
    ProfileRange,
    build_lateral_profile,
    render_lateral_profile,
)


def _mock_msgs():
    """Produce synthetic messages mimicking a short route."""
    from types import SimpleNamespace

    msgs = []
    mono = 0
    for i in range(100):
        mono += 50_000_000  # 50 ms between messages
        v = 15.0 + 5.0 * (i % 3)  # 15, 20, 25 m/s cycling
        k = 0.001 * (1 + i % 5)   # 0.001 to 0.005
        prob = 0.7 + 0.05 * (i % 5)  # 0.70 to 0.90
        roll = 0.02 * (i % 3)  # 0.0 to 0.04

        cm = SimpleNamespace(logMonoTime=mono, which=lambda: "carState")
        setattr(cm, "carState", SimpleNamespace(vEgo=v))
        msgs.append(cm)

        cm2 = SimpleNamespace(logMonoTime=mono + 1, which=lambda: "controlsState")
        setattr(cm2, "controlsState", SimpleNamespace(desiredCurvature=k, curvature=k * 0.95))
        msgs.append(cm2)

        cm3 = SimpleNamespace(logMonoTime=mono + 2, which=lambda: "modelV2")
        setattr(cm3, "modelV2", SimpleNamespace(laneLineProbs=[prob, prob, prob, prob]))
        msgs.append(cm3)

        cm4 = SimpleNamespace(logMonoTime=mono + 3, which=lambda: "liveParameters")
        setattr(cm4, "liveParameters", SimpleNamespace(roll=roll))
        msgs.append(cm4)

    return msgs


def test_build_lateral_profile_from_mock():
    profile = build_lateral_profile(_mock_msgs(), source="test:mock")
    assert isinstance(profile, LateralProfile)
    assert profile.source == "test:mock"
    assert profile.sample_count > 0
    assert profile.ego_speed.low > 0.0
    assert profile.ego_speed.high > profile.ego_speed.low
    assert profile.curvature.low >= 0.0
    assert profile.lane_confidence.low > 0.0
    assert profile.lane_confidence.high <= 1.05  # percentile float tolerance


def test_lateral_profile_roundtrip():
    profile = LateralProfile(
        source="test", sample_count=100,
        ego_speed=ProfileRange(10.0, 25.0),
        curvature=ProfileRange(0.0005, 0.003),
        lane_confidence=ProfileRange(0.5, 1.0),
        roll=ProfileRange(0.0, 0.05),
    )
    d = profile.to_dict()
    profile2 = LateralProfile.from_dict(d)
    assert profile2.ego_speed.low == 10.0
    assert profile2.curvature.high == 0.003


def test_render_lateral_profile():
    profile = LateralProfile(
        source="test", sample_count=50,
        ego_speed=ProfileRange(10.0, 25.0),
        curvature=ProfileRange(0.0005, 0.003),
        lane_confidence=ProfileRange(0.5, 1.0),
        roll=ProfileRange(0.0, 0.05),
    )
    text = render_lateral_profile(profile)
    assert "test" in text
    assert "ego_speed" in text
    assert "10.0000" in text
