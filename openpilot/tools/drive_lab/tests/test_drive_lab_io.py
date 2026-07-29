from pathlib import Path

from openpilot.tools.drive_lab.route_io import output_report


class DummyReport:
  def __init__(self):
    self.saved = False

  def to_dict(self):
    return {"z": 1, "a": 2}


def test_output_report_uses_renderer_and_saves_with_custom_save(tmp_path):
  report = DummyReport()
  path = tmp_path / "report.json"

  def save(_, p):
    Path(p).write_text("saved\n")

  rendered = output_report(report, json_output=False, renderer=lambda _: "rendered", output_path=path, save=save)

  assert rendered == "rendered"
  assert path.read_text() == "saved\n"


def test_output_report_writes_json_when_no_save_function(tmp_path):
  report = DummyReport()
  path = tmp_path / "report.json"

  rendered = output_report(report, json_output=True, renderer=lambda _: "rendered", output_path=path)

  assert rendered == '{\n  "z": 1,\n  "a": 2\n}'
  assert path.read_text() == '{\n  "a": 2,\n  "z": 1\n}\n'
