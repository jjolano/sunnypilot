import os


# BASEDIR is the repo root, not the openpilot package: it is joined with root-level paths
# such as "opendbc/car/torque_data/...". This file moved from common/ to openpilot/common/
# in the upstream layout migration, so it needs one more level up than before.
BASEDIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."))
