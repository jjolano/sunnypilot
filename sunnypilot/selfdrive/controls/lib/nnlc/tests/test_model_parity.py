import glob
import math
import os

import numpy as np
import pytest

from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.helpers import TORQUE_NN_MODEL_PATH
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.model import NNTorqueModel, PythonNNTorqueModel


CYTHON_AVAILABLE = NNTorqueModel is not PythonNNTorqueModel
REQUIRE_CYTHON = os.environ.get("REQUIRE_NNLC_CYTHON", "") == "1"
MODEL_PATHS = sorted(glob.glob(os.path.join(TORQUE_NN_MODEL_PATH, "*.json")))
assert MODEL_PATHS, f"No NNLC model JSON files found under {TORQUE_NN_MODEL_PATH}"


def _valid_lengths(input_size):
  lengths = {input_size}
  for length in (2, 3, 4):
    if 2 <= length <= input_size:
      lengths.add(length)
  return sorted(lengths)


class TestNNTorqueModelParity:

  @pytest.fixture(scope="class", autouse=True)
  def _require_cython(self):
    if not CYTHON_AVAILABLE:
      if REQUIRE_CYTHON:
        pytest.fail("REQUIRE_NNLC_CYTHON=1 but nnlc_model_pyx extension is unavailable")
      pytest.skip("nnlc_model_pyx extension is not built")

  @pytest.mark.parametrize("model_path", MODEL_PATHS)
  @pytest.mark.parametrize("zero_bias", [False, True])
  def test_parity(self, model_path, zero_bias):
    py_model = PythonNNTorqueModel(model_path, zero_bias=zero_bias)
    cy_model = NNTorqueModel(model_path, zero_bias=zero_bias)

    assert py_model.friction_override == cy_model.friction_override

    rng = np.random.default_rng(42)
    for length in _valid_lengths(py_model.input_size):
      for _ in range(5):
        inp = rng.normal(size=length).tolist()
        py_out = py_model.evaluate(inp)
        cy_out = cy_model.evaluate(inp)
        assert math.isclose(py_out, cy_out, rel_tol=1e-6, abs_tol=1e-6)

    full_length = rng.normal(size=py_model.input_size).astype(np.float32)
    for inp in (full_length.tolist(), tuple(full_length.tolist()), full_length):
      py_out = py_model.evaluate(inp)
      cy_out = cy_model.evaluate(inp)
      assert math.isclose(py_out, cy_out, rel_tol=1e-6, abs_tol=1e-6)

    x = ((full_length - py_model.input_mean) / py_model.input_std).astype(np.float32)
    np.testing.assert_allclose(py_model.forward(x), cy_model.forward(x), rtol=1e-6, atol=1e-6)
