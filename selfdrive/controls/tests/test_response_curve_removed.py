#!/usr/bin/env python3
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
LONGCONTROL = REPO_ROOT / "selfdrive" / "controls" / "lib" / "longcontrol.py"
LONGITUDINAL_PLANNER_SP = REPO_ROOT / "sunnypilot" / "selfdrive" / "controls" / "lib" / "longitudinal_planner.py"
PARAMS_KEYS = REPO_ROOT / "common" / "params_keys.h"
PARAMS_METADATA = REPO_ROOT / "sunnypilot" / "sunnylink" / "params_metadata.json"
CRUISE_SETTINGS = REPO_ROOT / "selfdrive" / "ui" / "sunnypilot" / "layouts" / "settings" / "cruise.py"
LONG_MASS_DRAG_LEARNER = REPO_ROOT / "sunnypilot" / "selfdrive" / "controls" / "lib" / "long_learned_mass_drag.py"
LONGCONTROL_EXT = REPO_ROOT / "sunnypilot" / "selfdrive" / "controls" / "lib" / "longcontrol_ext.py"


def test_longcontrol_does_not_apply_or_learn_response_curve_offsets():
  source = LONGCONTROL.read_text(encoding="utf-8")

  assert "get_response_offset" not in source
  assert "learn_response" not in source


def test_longcontrol_extension_has_no_response_curve_learner_or_params():
  assert not LONGCONTROL_EXT.exists()
  source = LONGCONTROL.read_text(encoding="utf-8")

  assert "ResponseCurveLearner" not in source
  assert "LongLearnedResponse" not in source


def test_longitudinal_mass_drag_learning_is_removed():
  assert not LONG_MASS_DRAG_LEARNER.exists()

  sources = (
    LONGCONTROL.read_text(encoding="utf-8"),
    LONGITUDINAL_PLANNER_SP.read_text(encoding="utf-8"),
    PARAMS_KEYS.read_text(encoding="utf-8"),
    PARAMS_METADATA.read_text(encoding="utf-8"),
    CRUISE_SETTINGS.read_text(encoding="utf-8"),
  )
  removed_tokens = (
    "LongLearnedMassDrag",
    "LongLearnedKForce",
    "LongLearnedCDrag",
    "RLSDynamicsEstimator",
    "mass_drag",
    "long_learned_mass_drag",
  )

  for source in sources:
    for token in removed_tokens:
      assert token not in source
