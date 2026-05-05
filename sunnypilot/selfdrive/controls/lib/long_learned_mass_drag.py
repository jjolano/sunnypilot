import numpy as np

MIN_VEGO = 10.0
MIN_POINTS_VALID = 50
KF_DEFAULT = 1.0
CDRAG_DEFAULT = 0.0
KF_MIN, KF_MAX = 0.5, 2.0
CDRAG_MIN, CDRAG_MAX = 0.0, 0.1


class RLSDynamicsEstimator:
  def __init__(self, forgetting_factor=0.995):
    self.lam = forgetting_factor
    # State: [k_force, c_drag]
    self.theta = np.array([[KF_DEFAULT], [CDRAG_DEFAULT]], dtype=np.float64)
    self.P = np.eye(2) * 10.0
    self.points = 0
    self._valid = False

  def update(self, v_ego, a_cmd, a_ego):
    if v_ego < MIN_VEGO:
      return

    # Regression model: a_ego = k_force * a_cmd - c_drag * v_ego^2
    phi = np.array([[a_cmd], [-v_ego ** 2]], dtype=np.float64)
    y = np.array([[a_ego]], dtype=np.float64)

    # RLS update
    P_phi = self.P @ phi
    denom = self.lam + phi.T @ P_phi
    K = P_phi / denom
    error = y - phi.T @ self.theta
    self.theta = self.theta + K @ error
    self.P = (self.P - K @ phi.T @ self.P) / self.lam

    # Sanity clamp and reset if needed
    k_force = float(self.theta[0, 0])
    c_drag = float(self.theta[1, 0])
    if not (KF_MIN <= k_force <= KF_MAX) or not (CDRAG_MIN <= c_drag <= CDRAG_MAX) or not np.isfinite(k_force) or not np.isfinite(c_drag):
      self.reset()
      return

    self.points += 1
    if self.points >= MIN_POINTS_VALID:
      self._valid = True

  def reset(self):
    self.theta = np.array([[KF_DEFAULT], [CDRAG_DEFAULT]], dtype=np.float64)
    self.P = np.eye(2) * 10.0
    self.points = 0
    self._valid = False

  def get_params(self):
    return float(self.theta[0, 0]), float(self.theta[1, 0])

  def is_valid(self):
    return self._valid
