from cereal import log

from openpilot.selfdrive.selfdrived.selfdrived import _comm_issue_event


EventName = log.OnroadEvent.EventName


class FakeSubMaster:
  def __init__(self, alive=True, freq_ok=True, valid=True):
    self.alive = alive
    self.freq_ok = freq_ok
    self.valid = valid

  def all_checks(self):
    return self.alive and self.freq_ok and self.valid

  def all_alive(self):
    return self.alive

  def all_freq_ok(self):
    return self.freq_ok

  def all_valid(self):
    return self.valid


def test_comm_issue_avg_freq_outside_startup_grace():
  sm = FakeSubMaster(freq_ok=False)

  assert _comm_issue_event(sm, suppress_avg_freq=False) == EventName.commIssueAvgFreq


def test_comm_issue_avg_freq_only_suppressed_during_startup_grace():
  sm = FakeSubMaster(freq_ok=False)

  assert _comm_issue_event(sm, suppress_avg_freq=True) is None


def test_comm_issue_not_alive_not_suppressed_during_startup_grace():
  sm = FakeSubMaster(alive=False, freq_ok=False, valid=False)

  assert _comm_issue_event(sm, suppress_avg_freq=True) == EventName.commIssue


def test_comm_issue_invalid_not_suppressed_during_startup_grace():
  sm = FakeSubMaster(freq_ok=False, valid=False)

  assert _comm_issue_event(sm, suppress_avg_freq=True) == EventName.commIssue
