from openpilot.sunnypilot.mapd.mapd_manager import build_mapd_download_paths, request_refresh_osm_location_data


class FakeParams:
  def __init__(self):
    self.values = {}

  def put(self, key, value):
    self.values[key] = value

  def put_bool(self, key, value):
    self.values[key] = bool(value)


class FakePubMaster:
  def __init__(self):
    self.messages = []

  def send(self, service, msg):
    self.messages.append((service, msg))


def test_build_mapd_download_paths_uses_v2_country_path():
  assert build_mapd_download_paths(["CA"], []) == ["nation.CA"]


def test_build_mapd_download_paths_uses_v2_us_state_path():
  assert build_mapd_download_paths(["US"], ["OH"]) == ["us_state.OH"]


def test_build_mapd_download_paths_keeps_filtered_us_state_path():
  assert build_mapd_download_paths([], ["OH"]) == ["us_state.OH"]


def test_build_mapd_download_paths_uses_country_for_all_us_states():
  assert build_mapd_download_paths(["US"], []) == ["nation.US"]


def test_request_refresh_osm_location_data_marks_json_download_paths(monkeypatch):
  fake_params = FakeParams()
  fake_mem_params = FakeParams()
  fake_pm = FakePubMaster()
  monkeypatch.setattr("openpilot.sunnypilot.mapd.mapd_manager.params", fake_params)
  monkeypatch.setattr("openpilot.sunnypilot.mapd.mapd_manager.mem_params", fake_mem_params)

  request_refresh_osm_location_data(fake_pm, ["US"], ["OH"])

  assert fake_mem_params.values["OSMDownloadLocations"] == {"paths": ["us_state.OH"], "pending": True}
  assert fake_pm.messages[0][0] == "mapdIn"
  assert fake_pm.messages[0][1].mapdIn.str == "us_state.OH"
