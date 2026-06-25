import math
import pytest

from openpilot.common.pid import PIDController, PythonPIDController


CYTHON_AVAILABLE = PIDController is not PythonPIDController


@pytest.fixture(params=[PythonPIDController, PIDController])
def Pid(request):
  return request.param


def _step(pid, error, **kwargs):
  return pid.update(error, error_rate=kwargs.pop("error_rate", 0.0), **kwargs)


class TestPIDCommon:
  def test_scalar_gains(self, Pid):
    pid = Pid(k_p=0.5, k_i=0.1, k_d=0.05, rate=100)
    control = pid.update(error=2.0, error_rate=0.5)

    assert math.isclose(pid.k_p, 0.5)
    assert math.isclose(pid.k_i, 0.1)
    assert math.isclose(pid.k_d, 0.05)
    assert math.isclose(pid.p, 1.0)
    assert math.isclose(pid.d, 0.025)
    assert math.isclose(pid.i, 0.002)
    assert math.isclose(pid.control, 1.027)
    assert math.isclose(control, pid.control)

  def test_breakpoint_gains(self, Pid):
    pid = Pid(
      k_p=[[0.0, 20.0], [1.0, 2.0]],
      k_i=[[0.0, 10.0, 30.0], [0.1, 0.2, 0.4]],
    )

    pid.speed = -5.0
    assert math.isclose(pid.k_p, 1.0)
    assert math.isclose(pid.k_i, 0.1)

    pid.speed = 5.0
    assert math.isclose(pid.k_p, 1.25)
    assert math.isclose(pid.k_i, 0.15)

    pid.speed = 10.0
    assert math.isclose(pid.k_p, 1.5)
    assert math.isclose(pid.k_i, 0.2)

    pid.speed = 20.0
    assert math.isclose(pid.k_p, 2.0)

    pid.speed = 30.0
    assert math.isclose(pid.k_i, 0.4)

    pid.speed = 100.0
    assert math.isclose(pid.k_p, 2.0)
    assert math.isclose(pid.k_i, 0.4)

  def test_nan_speed_parity_with_numpy_interp(self, Pid):
    scalar_pid = Pid(k_p=0.5, k_i=0.1)
    scalar_pid.speed = math.nan
    assert math.isclose(scalar_pid.k_p, 0.5)

    breakpoint_pid = Pid(k_p=[[0.0, 20.0], [1.0, 2.0]], k_i=0.1)
    breakpoint_pid.speed = math.nan
    assert math.isnan(breakpoint_pid.k_p)

  def test_duplicate_breakpoints_match_numpy_interp(self, Pid):
    pid = Pid(k_p=[[0.0, 0.0, 1.0], [1.0, 2.0, 3.0]], k_i=0.1)
    pid.speed = 0.0
    assert math.isclose(pid.k_p, 2.0)
    pid.speed = 0.5
    assert math.isclose(pid.k_p, 2.5)

  def test_freeze_integrator(self, Pid):
    pid = Pid(k_p=0.0, k_i=2.0, rate=100)

    pid.update(error=1.0, freeze_integrator=False)
    assert math.isclose(pid.i, 0.02)
    prev_i = pid.i

    pid.update(error=1.0, freeze_integrator=True)
    assert math.isclose(pid.i, prev_i)

  def test_clipping_and_windup(self, Pid):
    pid = Pid(k_p=0.0, k_i=1.0, pos_limit=1.0, neg_limit=-1.0, rate=2)

    for _ in range(20):
      pid.update(error=1.0)
    assert math.isclose(pid.control, 1.0)
    assert pid.control <= pid.pos_limit

    for _ in range(40):
      pid.update(error=-1.0)
    assert math.isclose(pid.control, -1.0)
    assert pid.control >= pid.neg_limit

  def test_set_limits_and_reset(self, Pid):
    pid = Pid(k_p=1.0, k_i=1.0, rate=100)
    pid.set_limits(1.0, -1.0)
    assert pid.pos_limit == 1.0
    assert pid.neg_limit == -1.0

    pid.update(error=100.0)
    assert pid.control == 1.0

    pid.reset()
    assert pid.p == 0.0
    assert pid.i == 0.0
    assert pid.d == 0.0
    assert pid.f == 0.0
    assert pid.control == 0

  def test_nan_control_matches_numpy_clip(self, Pid):
    pid = Pid(k_p=1.0, k_i=1.0, pos_limit=1.0, neg_limit=-1.0)
    assert math.isnan(pid.update(error=math.nan))

  def test_inverted_limits_match_numpy_clip(self, Pid):
    pid = Pid(k_p=1.0, k_i=0.0, pos_limit=-1.0, neg_limit=1.0)
    assert pid.update(error=0.0) == -1.0


@pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython extension not built")
class TestPIDParity:
  @staticmethod
  def _run_sequence(cls, k_p, k_i, k_d=0.0, pos_limit=1e308, neg_limit=-1e308, rate=100, steps=None):
    if steps is None:
      steps = [
        (1.0, 0.0, 0.0, 0.0, False),
        (1.0, 0.5, 5.0, 0.0, False),
        (-0.5, -0.2, 15.0, 0.1, True),
        (2.0, 1.0, 25.0, -0.1, False),
        (-5.0, 0.0, 100.0, 0.0, False),
        (10.0, -1.0, 5.0, 0.0, False),
      ] * 10
    pid = cls(k_p=k_p, k_i=k_i, k_d=k_d, pos_limit=pos_limit, neg_limit=neg_limit, rate=rate)
    for error, error_rate, speed, feedforward, freeze in steps:
      pid.update(error, error_rate=error_rate, speed=speed, feedforward=feedforward, freeze_integrator=freeze)
    return pid

  def _assert_parity(self, **kwargs):
    py = self._run_sequence(PythonPIDController, **kwargs)
    cy = self._run_sequence(PIDController, **kwargs)
    for attr in ("p", "i", "d", "f", "control"):
      assert math.isclose(getattr(py, attr), getattr(cy, attr), rel_tol=1e-12, abs_tol=1e-12), attr
    for attr in ("k_p", "k_i", "k_d"):
      assert math.isclose(getattr(py, attr), getattr(cy, attr), rel_tol=1e-12, abs_tol=1e-12), attr

  def test_scalar_parity(self):
    self._assert_parity(k_p=0.5, k_i=0.1, k_d=0.05)

  def test_breakpoint_parity(self):
    self._assert_parity(
      k_p=[[0.0, 20.0], [1.0, 2.0]],
      k_i=[[0.0, 10.0, 30.0], [0.1, 0.2, 0.4]],
      k_d=[[0.0, 50.0], [0.0, 0.3]],
    )

  def test_clipping_parity(self):
    self._assert_parity(k_p=0.0, k_i=1.0, pos_limit=0.5, neg_limit=-0.5)
