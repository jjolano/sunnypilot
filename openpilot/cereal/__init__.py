import os
import sys
import capnp
from importlib.resources import as_file, files

capnp.remove_import_hook()

with as_file(files("openpilot.cereal")) as fspath:
  CEREAL_PATH = fspath.as_posix()
  log = capnp.load(os.path.join(CEREAL_PATH, "log.capnp"))
  car = capnp.load(os.path.join(CEREAL_PATH, "car.capnp"))
  custom = capnp.load(os.path.join(CEREAL_PATH, "custom.capnp"))

# Transitional shim for the openpilot/ layout move. The pinned opendbc does:
#
#     try:
#       from cereal import car          # top-level `cereal`
#     except ImportError:
#       car = capnp.load(os.path.join(BASEDIR, "car.capnp"))
#
# (opendbc/car/structs.py, whose own TODO reads "remove car from cereal/__init__.py and
# always import from opendbc"). Before the move `cereal` was a top-level package, so opendbc
# reused the schema loaded here. Now it is `openpilot.cereal`, the fallback fires, and
# car.capnp gets loaded a second time under a different module -- capnp rejects that with
# "Duplicate ID @0x8e2af1e708af8b8d" and aborts the process, which breaks even `scons`.
#
# Aliasing the module keeps opendbc on the single already-loaded copy. DELETE THIS when the
# opendbc pin advances to upstream's (d6b9c1a): that revision drops the fallback, and
# upstream's cereal no longer exports `car` at all.
sys.modules.setdefault("cereal", sys.modules[__name__])
