from openpilot.sunnypilot.livedelay.helpers import get_lat_delay


class Params:
  def __init__(self, live: bool):
    self.live = live

  def get_bool(self, _key):
    return self.live

  def get(self, _key, return_default=False):
    return 0.3


def test_lat_delay_selects_live_or_fixed_value():
  assert get_lat_delay(Params(live=True), 0.12) == 0.12
  assert get_lat_delay(Params(live=False), 0.12) == 0.3
