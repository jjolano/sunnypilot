#!/usr/bin/env python3
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
LONGCONTROL = REPO_ROOT / "selfdrive" / "controls" / "lib" / "longcontrol.py"
LONGCONTROL_EXT = REPO_ROOT / "sunnypilot" / "selfdrive" / "controls" / "lib" / "longcontrol_ext.py"


def test_longcontrol_does_not_apply_or_learn_response_curve_offsets():
  source = LONGCONTROL.read_text(encoding="utf-8")

  assert "get_response_offset" not in source
  assert "learn_response" not in source


def test_longcontrol_extension_has_no_response_curve_learner_or_params():
  source = LONGCONTROL_EXT.read_text(encoding="utf-8")

  assert "ResponseCurveLearner" not in source
  assert "LongLearnedResponse" not in source
