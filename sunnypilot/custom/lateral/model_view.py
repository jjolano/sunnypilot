"""Drift-tolerant snapshot of the modelV2 fields the torque lateral evidence reads.

The torque-jerk evidence (``latcontrol_torque_ext_base_v1`` and its consolidated twin
``parameter_orchestrator.TorqueModelEvidence``) previously read ``model_v2.orientation.x``
and ``model_v2.acceleration.y`` directly. A comma schema rename/removal on an upstream sync
would raise ``AttributeError`` inside ``controlsd`` rather than fail closed. This view is the
single place that knows those field shapes: a missing/renamed field degrades to ``valid=False``
(the controller then skips the jerk feed-forward and keeps stock behavior) instead of crashing.

Scoped to the fields actually consumed. Other custom modelV2 readers already fail closed
(``curve_evidence.vision_controller`` try/except, ``curve_traffic_advisor`` getattr path) or
take pre-extracted scalars (``model_trust``), so they do not go through here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N


@dataclass(frozen=True)
class ModelView:
  valid: bool = False
  acceleration_y: tuple[float, ...] = ()

  @classmethod
  def from_msg(cls, model_v2: Any) -> ModelView:
    # No `or`/truthiness on the arrays: modelV2 fields are numpy/capnp sequences whose
    # bool() is ambiguous. Guard on None + len only, and swallow any shape surprise.
    if model_v2 is None:
      return cls()
    ori_x = getattr(getattr(model_v2, "orientation", None), "x", None)
    accel_y = getattr(getattr(model_v2, "acceleration", None), "y", None)
    try:
      valid = ori_x is not None and len(ori_x) >= CONTROL_N
    except TypeError:
      valid = False
    try:
      accel = () if accel_y is None else tuple(accel_y)
    except TypeError:
      accel = ()
    return cls(valid=valid, acceleration_y=accel)
