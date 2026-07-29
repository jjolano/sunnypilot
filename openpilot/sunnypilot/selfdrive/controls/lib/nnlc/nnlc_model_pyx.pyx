# distutils: language = c++
# cython: language_level = 3
import numpy as np

from openpilot.selfdrive.modeld.parse_model_outputs import safe_exp


def _forward(object x, list layers):
  cdef object W, b
  cdef str activation

  for W, b, activation in layers:
    x = x.dot(W) + b
    if activation == 'sigmoid':
      x = 1.0 / (1.0 + safe_exp(-x))
    elif activation != 'identity':
      raise ValueError(f"Unknown activation: {activation}")

  return x


def _evaluate(object input_array, int input_size, object input_mean, object input_std, list layers):
  cdef int in_len = len(input_array)

  if in_len != input_size:
    if in_len >= 2:
      input_array = input_array + [0] * (input_size - in_len)
    else:
      raise ValueError(f"Input array length {in_len} must be length 2 or greater")

  cdef object x = np.array(input_array, dtype=np.float32)
  x = (x - input_mean) / input_std

  return float(_forward(x, layers)[0, 0])
