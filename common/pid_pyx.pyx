# distutils: language = c++
# cython: language_level = 3
from numbers import Number
import bisect


cdef inline double _interp_speed(double x, object gain) except *:
  cdef object xp = gain[0]
  cdef object fp = gain[1]
  cdef int n = len(xp)

  if n == 1:
    return <double>fp[0]
  if x != x:
    return x

  if x < <double>xp[0]:
    return <double>fp[0]
  if x > <double>xp[-1]:
    return <double>fp[-1]

  cdef int idx = bisect.bisect_right(xp, x) - 1
  if idx < 0:
    idx = 0
  if idx >= n - 1:
    return <double>fp[-1]

  cdef double x0 = <double>xp[idx]
  cdef double x1 = <double>xp[idx + 1]
  cdef double denom = x1 - x0
  if denom == 0.0:
    return <double>fp[idx]
  return <double>fp[idx] + (x - x0) * (<double>fp[idx + 1] - <double>fp[idx]) / denom


cdef inline object _clip_scalar(object value, object lower, object upper):
  if value != value or lower != lower or upper != upper:
    return float("nan")
  if lower > upper:
    return upper
  if value < lower:
    return lower
  if value > upper:
    return upper
  return value


cdef class PIDController:
  cdef public object _k_p
  cdef public object _k_i
  cdef public object _k_d
  cdef public object pos_limit
  cdef public object neg_limit
  cdef public double i_dt
  cdef public double speed
  cdef public object p
  cdef public object i
  cdef public object d
  cdef public object f
  cdef public object control

  def __init__(self, k_p, k_i, k_d=0., pos_limit=1e308, neg_limit=-1e308, rate=100):
    self._k_p = [[0], [k_p]] if isinstance(k_p, Number) else k_p
    self._k_i = [[0], [k_i]] if isinstance(k_i, Number) else k_i
    self._k_d = [[0], [k_d]] if isinstance(k_d, Number) else k_d

    self.pos_limit = pos_limit
    self.neg_limit = neg_limit
    self.i_dt = 1.0 / rate
    self.speed = 0.0
    self.reset()

  @property
  def k_p(self):
    return _interp_speed(self.speed, self._k_p)

  @property
  def k_i(self):
    return _interp_speed(self.speed, self._k_i)

  @property
  def k_d(self):
    return _interp_speed(self.speed, self._k_d)

  def reset(self):
    self.p = 0.0
    self.i = 0.0
    self.d = 0.0
    self.f = 0.0
    self.control = 0

  def set_limits(self, pos_limit, neg_limit):
    self.pos_limit = pos_limit
    self.neg_limit = neg_limit

  def update(self, error, error_rate=0.0, speed=0.0, feedforward=0., freeze_integrator=False):
    self.speed = speed
    self.p = self.k_p * float(error)
    self.d = self.k_d * error_rate
    self.f = feedforward

    if not freeze_integrator:
      i = self.i + self.k_i * self.i_dt * error

      # Don't allow windup if already clipping
      test_control = self.p + i + self.d + self.f
      i_upperbound = self.i if test_control > self.pos_limit else self.pos_limit
      i_lowerbound = self.i if test_control < self.neg_limit else self.neg_limit
      self.i = _clip_scalar(i, i_lowerbound, i_upperbound)

    control = self.p + self.i + self.d + self.f
    self.control = _clip_scalar(control, self.neg_limit, self.pos_limit)
    return self.control
