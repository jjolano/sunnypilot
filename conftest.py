import contextlib
import gc
import os
import resource
import pytest

from openpilot.common.prefix import OpenpilotPrefix
from openpilot.system.manager import manager
from openpilot.common.hardware import TICI, HARDWARE

# these are heavy CI-only tests, invoked explicitly in .github/workflows/tests.yaml
collect_ignore = [
  "openpilot/selfdrive/test/process_replay/test_processes.py",
  "openpilot/selfdrive/test/process_replay/test_regen.py",

  "openpilot/tools/sim/",

  # tinygrad JIT has process-global state. Other test files import modeld → tinygrad,
  # which corrupts JIT captures for test_warp.py in the same process. Run separately in CI.
  "openpilot/sunnypilot/modeld_v2/tests/test_warp.py",
]


def pytest_sessionstart(session):
  # TODO: fix tests and enable test order randomization
  if session.config.pluginmanager.hasplugin('randomly'):
    session.config.option.randomly_reorganize = False


def _limit_core_dumps():
  """Keep crashing test subprocesses from dumping cores into the working tree.

  Tests spawn native processes (locationd, managed processes under process_replay) with the
  repo root as CWD. With the usual `kernel.core_pattern=core` and `ulimit -c unlimited`, one
  abort writes a ~126 MB core.<pid> next to the source. 37 of them accumulated unnoticed
  between 2026-07-07 and 2026-07-18 — 5.4 GB — because they are gitignored and so never
  showed up in git status.

  rlimits are inherited across fork/exec, so setting this once here covers every child.
  Set OPENPILOT_TEST_CORE_DUMPS=1 to keep cores when you are actually debugging a crash.
  """
  if os.environ.get("OPENPILOT_TEST_CORE_DUMPS") == "1":
    return
  try:
    _, hard = resource.getrlimit(resource.RLIMIT_CORE)
    resource.setrlimit(resource.RLIMIT_CORE, (0, hard))
  except (ValueError, OSError):
    pass  # not fatal: worst case we are back to today's behavior


_limit_core_dumps()


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_call(item):
  # ensure we run as a hook after capturemanager's
  if item.get_closest_marker("nocapture") is not None:
    capmanager = item.config.pluginmanager.getplugin("capturemanager")
    with capmanager.global_and_fixture_disabled():
      yield
  else:
    yield


@contextlib.contextmanager
def clean_env():
  starting_env = dict(os.environ)
  yield
  os.environ.clear()
  os.environ.update(starting_env)


@pytest.fixture(scope="function", autouse=True)
def openpilot_function_fixture(request):
  with clean_env():
    # setup a clean environment for each test
    with OpenpilotPrefix(shared_download_cache=request.node.get_closest_marker("shared_download_cache") is not None) as prefix:
      prefix = os.environ["OPENPILOT_PREFIX"]

      yield

      # ensure the test doesn't change the prefix
      assert "OPENPILOT_PREFIX" in os.environ and prefix == os.environ["OPENPILOT_PREFIX"]

    # cleanup any started processes
    manager.manager_cleanup()

    # some processes disable gc for performance, re-enable here
    if not gc.isenabled():
      gc.enable()
      gc.collect()

# If you use setUpClass, the environment variables won't be cleared properly,
# so we need to hook both the function and class pytest fixtures
@pytest.fixture(scope="class", autouse=True)
def openpilot_class_fixture():
  with clean_env():
    yield


@pytest.fixture(scope="function")
def tici_setup_fixture(request, openpilot_function_fixture):
  """Ensure a consistent state for tests on-device. Needs the openpilot function fixture to run first."""
  if 'skip_tici_setup' in request.keywords:
    return
  HARDWARE.initialize_hardware()
  HARDWARE.set_power_save(False)
  os.system("pkill -9 -f athena")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
  skipper = pytest.mark.skip(reason="Skipping tici test on PC")
  for item in items:
    if "tici" in item.keywords:
      if not TICI:
        item.add_marker(skipper)
      else:
        item.fixturenames.append('tici_setup_fixture')

    if "xdist_group_class_property" in item.keywords:
      class_property_name = item.get_closest_marker('xdist_group_class_property').args[0]
      class_property_value = getattr(item.cls, class_property_name)
      item.add_marker(pytest.mark.xdist_group(class_property_value))
