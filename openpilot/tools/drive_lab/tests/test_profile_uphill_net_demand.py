from __future__ import annotations

import numpy as np
import pytest

from openpilot.tools.drive_lab.profile_uphill_net_demand import analyze_npz


def test_profile_reports_known_coast_zero_and_no_go_without_shift_labels(tmp_path) -> None:
  t = np.arange(0.0, 80.0, 0.05)
  pitch = 0.01 + 0.06 * (t / t[-1])
  a_ego = -8.0 * pitch + 0.32
  long_active = t >= 45.0
  path = tmp_path / "route.npz"
  np.savez(
    path,
    cs_t=t,
    cs_v=np.full_like(t, 20.0),
    cs_a=a_ego,
    cs_gp=np.zeros_like(t),
    cs_bp=np.zeros_like(t),
    cc_t=t,
    cc_pitch=pitch,
    cc_la=long_active,
    cc_ac=np.full_like(t, 0.6),
  )

  report = analyze_npz(path)

  assert report["fit"]["ready"] is True
  assert report["fit"]["pitch_zero_rad"] == pytest.approx(0.04, abs=1e-3)
  assert report["apply_go"] is False
  assert "no labeled steep-climb shift/no-shift corpus" in report["no_go_reasons"]
