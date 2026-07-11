from types import SimpleNamespace

from openpilot.tools.drive_lab.analyze_longitudinal_lateral_route import AnalysisWindow, extract_rows


class FakeMsg(SimpleNamespace):
  def which(self):
    return self.kind


def msg(kind: str, t_s: float, **payload):
  return FakeMsg(kind=kind, logMonoTime=int(t_s * 1e9), **{kind: SimpleNamespace(**payload)})


def test_extract_rows_uses_controls_state_as_the_sampling_clock():
  rows = extract_rows([
    msg("carState", 0.0, vEgo=10.0),
    msg("controlsState", 0.01, desiredCurvature=0.001),
    msg("modelV2", 0.02, action=SimpleNamespace(desiredAcceleration=-0.1)),
    msg("carState", 0.03, vEgo=11.0),
    msg("controlsState", 0.04, desiredCurvature=0.002),
  ], AnalysisWindow(0.0, 1.0, "test"))

  assert [row["msg_type"] for row in rows] == ["controlsState", "controlsState"]
  assert [row["v_ego"] for row in rows] == [10.0, 11.0]
  assert [row["model_action_desired_accel"] for row in rows] == ["", -0.1]
