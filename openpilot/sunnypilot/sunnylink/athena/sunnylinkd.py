#!/usr/bin/env python3
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from __future__ import annotations

import base64
import errno
import gzip
import json
import os
import math
import ssl
import threading
import time
from typing import Any

from functools import partial
from openpilot.system.athena.rpc import dispatcher
from openpilot.common.params import Params, ParamKeyType
from openpilot.common.realtime import set_core_affinity
from openpilot.common.swaglog import cloudlog
from openpilot.common.hardware.hw import Paths
from openpilot.system.athena.athenad import ws_send, jsonrpc_handler, \
  recv_queue, UploadQueueCache, upload_queue, cur_upload_items, backoff, ws_manage, log_handler, start_local_proxy_shim, upload_handler, stat_handler
from websocket import (ABNF, WebSocket, WebSocketException, WebSocketTimeoutException,
                       create_connection, WebSocketConnectionClosedException)

import openpilot.cereal.messaging as messaging
from openpilot.sunnypilot.models.default_model import DEFAULT_MODEL
from openpilot.sunnypilot.selfdrive.car.sync_sunnylink_params import update_car_list_param
from openpilot.sunnypilot.sunnylink.api import SunnylinkApi
from openpilot.sunnypilot.sunnylink.utils import sunnylink_need_register, sunnylink_ready, get_param_as_byte, save_param_from_base64_encoded_string
from openpilot.sunnypilot.sunnylink.capabilities import generate_capabilities, CAPABILITY_LABELS
from openpilot.sunnypilot.sunnylink.tools.generate_settings_schema import generate_schema

SUNNYLINK_ATHENA_HOST = os.getenv('SUNNYLINK_ATHENA_HOST', 'wss://athena.sunnylink.ai')
HANDLER_THREADS = int(os.getenv('HANDLER_THREADS', "4"))
LOCAL_PORT_WHITELIST = {8022}
SUNNYLINK_LOG_ATTR_NAME = "user.sunny.upload"
SUNNYLINK_RECONNECT_TIMEOUT_S = 70  # FYI changing this will also would require a change on sidebar.cc
DISALLOW_LOG_UPLOAD = threading.Event()

params = Params()

# Parameters that should never be remotely modified
BLOCKED_PARAMS = {
  "AdbEnabled",
  "CompletedSunnylinkConsentVersion",
  "CompletedTrainingVersion",
  "GithubUsername",  # Could grant SSH access
  "GithubSshKeys",   # Direct SSH key injection
  "HasAcceptedTerms",
  "HasAcceptedTermsSP",
  "AlphaLongitudinalEnabled",   # Safety-critical control mode; needs local/offroad confirmation
  "JoystickDebugMode",          # Can stop controls and start joystickd
  "LateralManeuverMode",        # Test-only driving mode
  "LongitudinalManeuverMode",   # Test-only driving mode
  "LiveTorqueSpeedAdaptiveParams",  # Hidden live steering profile; generated locally only
  "RollCompGainParams",       # Hidden live roll-compensation profile; generated locally only
  "OnroadCycleRequested",      # Prevent remote cycle trigger
  "ParamsVersion",         # Device-managed version counter
  "SshEnabled",           # Remote shell access must stay local-only
}

# Safety-critical torque/control settings that require the device to be offroad.
# These are defense-in-depth checks on top of metadata policy (blocked/attestation/
# min/max/options). Writes are rejected unless IsOffroad is true.
SAFETY_CRITICAL_REMOTE_GATED_PARAMS = {
  "TorqueParamsOverrideEnabled",
  "TorqueParamsOverrideLatAccelFactor",
  "TorqueParamsOverrideFriction",
  "LiveTorqueParamsToggle",
  "LiveTorqueParamsRelaxedToggle",
  "LiveTorqueSpeedAdaptiveMode",
  "LiveTorqueLowSpeedShadow",
  "RollCompGainMode",
  "EnforceTorqueControl",
  "TorqueControlTune",
  "CustomTorqueParams",
  "NeuralNetworkLateralControl",
  "CustomLateralDemandEnabled",
  "LateralPreviewAssistMode",
  "LaneCenteringAssistEnabled",
}

# Manual override state and values must only be activated while CustomTorqueParams is
# enabled. We allow the dependency to be satisfied by the current param value or by
# an atomic CustomTorqueParams=true write in the same saveParams transaction.
DEPENDS_ON_CUSTOM_TORQUE_PARAMS = {
  "TorqueParamsOverrideEnabled",
  "TorqueParamsOverrideLatAccelFactor",
  "TorqueParamsOverrideFriction",
}

# Service/action toggles and request params that must only be changed locally,
# never remotely. These control on-device services or trigger device-side actions.
SERVICE_ACTION_WRITE_DENY_SET = {
  "EnableCopyparty",
  "EnableGithubRunner",
  "EnableSunnylinkUploader",
  "EnableTailscale",
  "TailscaleLoginRequested",
  "TailscaleLogoutRequested",
  "TailscaleInstallRequested",
}

# Sensitive/DONT_LOG param keys that must never be returned by getParams.
SENSITIVE_READ_DENY_SET = {
  "AccessToken",
  "LivestreamEncoderBitrate",
  "LiveTorqueParameters",
  "LiveTorqueSpeedAdaptiveParams",
  "RollCompGainParams",
  "SecOCKey",
  "TailscaleAuthURL",
}


def remote_read_allowed(key: str) -> bool:
  """Return True only if the key may be read remotely.

  Default-deny: only keys with a non-blocked SETTINGS_POLICY entry are allowed.
  BLOCKED_PARAMS, sensitive/DONT_LOG keys, and service/action keys are rejected.
  """
  if key in BLOCKED_PARAMS:
    return False
  if key in SENSITIVE_READ_DENY_SET:
    return False
  if key in SERVICE_ACTION_WRITE_DENY_SET:
    return False
  policy = SETTINGS_POLICY.get(key)
  if policy is None or policy.get("blocked"):
    return False
  return True


def remote_write_policy(key: str) -> dict[str, Any] | None:
  """Return the SETTINGS_POLICY entry for the key if remote writes are allowed.

  Default-deny: blocked params, service/action params, and unlisted/unknown keys
  return None. Known schema keys return their policy dict for attestation,
  offroad, range, and option validation.
  """
  if key in BLOCKED_PARAMS:
    return None
  if key in SERVICE_ACTION_WRITE_DENY_SET:
    return None
  policy = SETTINGS_POLICY.get(key)
  if policy is None:
    return None
  if policy.get("blocked"):
    return None
  return policy


def _decode_param_value(value: str, compression: bool) -> str:
  raw = base64.b64decode(value, validate=True)
  if compression:
    raw = gzip.decompress(raw)
  return raw.decode("utf-8")


def _option_value_matches(option_value, candidate: str) -> bool:
  if isinstance(option_value, bool):
    if candidate.lower() in {"true", "1"}:
      return option_value is True
    if candidate.lower() in {"false", "0"}:
      return option_value is False
    return False

  if isinstance(option_value, (int, float)) and not isinstance(option_value, bool):
    try:
      candidate_num = float(candidate)
    except ValueError:
      return False
    return math.isfinite(candidate_num) and candidate_num == float(option_value)

  return str(option_value) == candidate


def _decoded_value_matches_policy(decoded_value: str, policy: dict[str, Any]) -> bool:
  if "min" in policy or "max" in policy:
    try:
      numeric_value = float(decoded_value)
    except ValueError:
      return False
    if not math.isfinite(numeric_value):
      return False
    if "min" in policy and numeric_value < float(policy["min"]):
      return False
    if "max" in policy and numeric_value > float(policy["max"]):
      return False

  if "options" in policy:
    option_values = [opt.get("value") for opt in policy.get("options", []) if isinstance(opt, dict) and "value" in opt]
    if not any(_option_value_matches(option_value, decoded_value) for option_value in option_values):
      return False

  return True


def _collect_settings_policy() -> dict[str, dict[str, Any]]:
  schema = generate_schema()
  policy: dict[str, dict[str, Any]] = {}

  def record_item_policy(key: str, item_policy: dict[str, Any]) -> None:
    current = policy.setdefault(str(key), {})
    current["blocked"] = bool(current.get("blocked")) or bool(item_policy.get("blocked"))
    current["attestation_required"] = bool(current.get("attestation_required")) or bool(item_policy.get("attestation_required"))
    for field in ("min", "max", "options"):
      if field in item_policy:
        current[field] = item_policy[field]

  def walk_item(item: dict, inherited_attestation: bool, remote_configurable: bool) -> None:
    item_attestation = inherited_attestation or bool(item.get("requires_attestation"))
    item_policy: dict[str, Any] = {
      "blocked": bool(item.get("blocked")) or (not remote_configurable),
      "attestation_required": item_attestation,
    }
    for field in ("min", "max", "options"):
      if field in item:
        item_policy[field] = item[field]

    key = item.get("key")
    if key:
      record_item_policy(str(key), item_policy)

    for sub_item in item.get("sub_items", []) or []:
      if isinstance(sub_item, dict):
        walk_item(sub_item, item_attestation, remote_configurable)

  def walk_container(node: dict, inherited_attestation: bool = False, parent_remote_configurable: bool = True) -> None:
    if not isinstance(node, dict):
      return

    current_attestation = inherited_attestation or bool(node.get("attestation_required"))
    current_remote_configurable = parent_remote_configurable
    if "remote_configurable" in node:
      current_remote_configurable = bool(node.get("remote_configurable"))

    for item in node.get("items", []) or []:
      if isinstance(item, dict):
        walk_item(item, current_attestation, current_remote_configurable)

    for section in node.get("sections", []) or []:
      if isinstance(section, dict):
        walk_container(section, current_attestation, current_remote_configurable)

    for sub_panel in node.get("sub_panels", []) or []:
      if isinstance(sub_panel, dict):
        walk_container(sub_panel, current_attestation, current_remote_configurable)

  for panel in schema.get("panels", []) or []:
    if isinstance(panel, dict):
      walk_container(panel)

  vehicle_settings = schema.get("vehicle_settings", {})
  if isinstance(vehicle_settings, dict):
    for vehicle_setting in vehicle_settings.values():
      if isinstance(vehicle_setting, dict):
        walk_container(vehicle_setting)
      elif isinstance(vehicle_setting, list):
        for item in vehicle_setting:
          if isinstance(item, dict):
            walk_item(item, False, True)

  return policy


SETTINGS_POLICY = _collect_settings_policy()


def _is_remote_value_true(value: str, compression: bool) -> bool:
  """Decode a base64 (optionally gzipped) remote bool value and return whether it is true."""
  try:
    decoded = _decode_param_value(value, compression)
  except Exception:
    return False
  return decoded.lower() in {"1", "true"}


def _custom_torque_enabled_after_valid_request(params_to_update: dict[str, str], compression: bool,
                                               attested_keys: set[str], offroad: bool) -> bool:
  """Return whether this transaction validly satisfies the CustomTorqueParams dependency."""
  if params.get_bool("CustomTorqueParams"):
    return True
  value = params_to_update.get("CustomTorqueParams")
  if value is None or not offroad or not _torque_settings_allowed():
    return False

  policy = remote_write_policy("CustomTorqueParams")
  if policy is None:
    return False
  if policy.get("attestation_required") and "CustomTorqueParams" not in attested_keys:
    return False
  try:
    decoded_value = _decode_param_value(value, compression)
  except Exception:
    return False
  if not _decoded_value_matches_policy(decoded_value, policy):
    return False
  return decoded_value.strip().lower() in {"1", "true"}


def _torque_settings_allowed() -> bool:
  """Return True unless capability data explicitly indicates torque steering is not allowed.

  When no CarParams or CarPlatformBundle is available yet, we default to True so
  pre-fingerprint devices can still receive torque settings safely gated by other
  checks (offroad, attestation, etc.).
  """
  if params.get("CarParamsPersistent") is None and params.get("CarPlatformBundle") is None:
    return True
  try:
    return bool(generate_capabilities().get("torque_allowed"))
  except Exception:
    cloudlog.exception("sunnylinkd._torque_settings_allowed.exception")
    return True


def handle_long_poll(ws: WebSocket, exit_event: threading.Event | None) -> None:
  cloudlog.info("sunnylinkd.handle_long_poll started")
  sm = messaging.SubMaster(['deviceState'])
  end_event = threading.Event()
  comma_prime_cellular_end_event = threading.Event()

  threads = [
              threading.Thread(target=ws_manage, args=(ws, end_event), name='ws_manage'),
              threading.Thread(target=ws_recv, args=(ws, end_event), name='ws_recv'),
              threading.Thread(target=ws_send, args=(ws, end_event), name='ws_send'),
              threading.Thread(target=ws_ping, args=(ws, end_event), name='ws_ping'),
              threading.Thread(target=upload_handler, args=(end_event,), name='upload_handler'),
              threading.Thread(target=sunny_log_handler, args=(end_event, comma_prime_cellular_end_event), name='log_handler'),
              threading.Thread(target=stat_handler, args=(end_event, Paths.stats_sp_root(), True), name='stat_handler'),
            ] + [
              threading.Thread(target=jsonrpc_handler, args=(end_event, partial(startLocalProxy, end_event),), name=f'worker_{x}')
              for x in range(HANDLER_THREADS)
            ]

  for thread in threads:
    thread.start()
  try:
    while not end_event.wait(0.1):
      if not sunnylink_ready(params):
        cloudlog.warning("Exiting sunnylinkd.handle_long_poll as SunnylinkEnabled is False")
        break

      sm.update(0)
      if exit_event is not None and exit_event.is_set():
        end_event.set()
        comma_prime_cellular_end_event.set()

      prime_type = params.get("PrimeType") or 0
      metered = sm['deviceState'].networkMetered

      if DISALLOW_LOG_UPLOAD.is_set() and not comma_prime_cellular_end_event.is_set():
        cloudlog.debug("sunnylinkd.handle_long_poll: DISALLOW_LOG_UPLOAD, setting comma_prime_cellular_end_event")
        comma_prime_cellular_end_event.set()
      elif metered and int(prime_type) > 2:
        cloudlog.debug(f"sunnylinkd.handle_long_poll: PrimeType({prime_type}) > 2 and networkMetered({metered})")
        comma_prime_cellular_end_event.set()
      elif comma_prime_cellular_end_event.is_set() and not DISALLOW_LOG_UPLOAD.is_set():
        cloudlog.debug(
          f"sunnylinkd.handle_long_poll: comma_prime_cellular_end_event is set and not PrimeType({prime_type}) > 2 or not networkMetered({metered})")
        comma_prime_cellular_end_event.clear()
  finally:
    end_event.set()
    comma_prime_cellular_end_event.set()
    for thread in threads:
      cloudlog.debug(f"sunnylinkd athena.joining {thread.name}")
      thread.join()
      cloudlog.debug(f"sunnylinkd athena.joined {thread.name}")


def ws_recv(ws: WebSocket, end_event: threading.Event) -> None:
  last_ping = int(time.monotonic() * 1e9)
  while not end_event.is_set():
    try:
      opcode, data = ws.recv_data(control_frame=True)
      if opcode in (ABNF.OPCODE_TEXT, ABNF.OPCODE_BINARY):
        if opcode == ABNF.OPCODE_TEXT:
          data = data.decode("utf-8")
        recv_queue.put_nowait(data)
        cloudlog.debug(f"sunnylinkd.ws_recv.recv {data}")
      elif opcode in (ABNF.OPCODE_PING, ABNF.OPCODE_PONG):
        cloudlog.debug("sunnylinkd.ws_recv.pong")
        last_ping = int(time.monotonic() * 1e9)
        Params().put("LastSunnylinkPingTime", last_ping, block=True)
    except WebSocketTimeoutException:
      ns_since_last_ping = int(time.monotonic() * 1e9) - last_ping
      if ns_since_last_ping > SUNNYLINK_RECONNECT_TIMEOUT_S * 1e9:
        cloudlog.warning("sunnylinkd.ws_recv.timeout")
        end_event.set()
    except Exception as e:
      if isinstance(e, WebSocketConnectionClosedException):
        cloudlog.warning(f"sunnylinkd.ws_recv.{type(e).__name__}")
      else:
        cloudlog.exception("sunnylinkd.ws_recv.exception")
      end_event.set()


def ws_ping(ws: WebSocket, end_event: threading.Event) -> None:
  ws.ping()  # Send the first ping
  while not end_event.wait(SUNNYLINK_RECONNECT_TIMEOUT_S * 0.7):  # Sleep about 70% before a timeout
    try:
      ws.ping()
      cloudlog.debug("sunnylinkd.ws_recv.ws_ping: Pinging")
    except Exception:
      cloudlog.exception("sunnylinkd.ws_ping.exception")
      end_event.set()
  cloudlog.debug("sunnylinkd.ws_ping.end_event is set, exiting ws_ping thread")


def sunny_log_handler(end_event: threading.Event, comma_prime_cellular_end_event: threading.Event) -> None:
  while not end_event.wait(0.1):
    if not comma_prime_cellular_end_event.is_set():
      log_handler(comma_prime_cellular_end_event, SUNNYLINK_LOG_ATTR_NAME)
  comma_prime_cellular_end_event.set()


@dispatcher.add_method
def toggleLogUpload(enabled: bool):
  DISALLOW_LOG_UPLOAD.clear() if enabled and DISALLOW_LOG_UPLOAD.is_set() else DISALLOW_LOG_UPLOAD.set()


@dispatcher.add_method
def getParamsAllKeys() -> list[str]:
  keys: list[str] = [k.decode('utf-8') for k in Params().all_keys() if remote_read_allowed(k.decode('utf-8'))]
  return keys


@dispatcher.add_method
def getParamsMetadata() -> str:
  """Return settings_ui.json + live capabilities as gzip-compressed, base64-encoded string.

  Reads settings_ui.json, injects live capabilities from CarParams, compresses,
  and returns. Single RPC for the frontend to get the complete settings UI and
  runtime capabilities.
  """
  try:
    schema = generate_schema()
    schema["capabilities"] = generate_capabilities()
    schema["capability_labels"] = CAPABILITY_LABELS
    schema["default_model"] = DEFAULT_MODEL
    raw = json.dumps(schema, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(gzip.compress(raw)).decode("utf-8")
  except Exception:
    cloudlog.exception("sunnylinkd.getParamsMetadata.exception")
    raise


@dispatcher.add_method
def getParams(params_keys: list[str], compression: bool = False) -> str | dict[str, str]:
  params = Params()
  available_keys: list[str] = [k.decode('utf-8') for k in Params().all_keys()]

  try:
    zero_values: dict[int, bytes] = {
      ParamKeyType.STRING.value: b"",
      ParamKeyType.BOOL.value: b"0",
      ParamKeyType.INT.value: b"0",
      ParamKeyType.FLOAT.value: b"0.0",
      ParamKeyType.TIME.value: b"",
      ParamKeyType.JSON.value: b"{}",
      ParamKeyType.BYTES.value: b"",
    }

    param_keys_validated = [key for key in params_keys if key in available_keys and remote_read_allowed(key)]
    params_dict: dict[str, list[dict[str, str | bool | int]]] = {"params": []}
    for key in param_keys_validated:
      value = get_param_as_byte(key)
      if value is None:
        value = get_param_as_byte(key, get_default=True)
      if value is None:
        param_type = params.get_type(key)
        value = zero_values.get(param_type.value, b"")

      params_dict["params"].append({
        "key": key,
        "value": base64.b64encode(gzip.compress(value) if compression else value).decode('utf-8'),
        "type": int(params.get_type(key).value),
        "is_compressed": compression
      })

    response = {str(param.get('key')): str(param.get('value')) for param in params_dict.get("params", [])}
    response |= {"params": json.dumps(params_dict.get("params", []))} # Upcoming for settings v1
    return response

  except Exception as e:
    cloudlog.exception("sunnylinkd.getParams.exception", e)
    raise


@dispatcher.add_method
def saveParams(params_to_update: dict[str, str], compression: bool = False, attested_params: list[str] | dict[str, bool] | None = None) -> None:
  attested_keys = set(attested_params if isinstance(attested_params, (list, set, tuple)) else [])
  if isinstance(attested_params, dict):
    attested_keys |= {key for key, value in attested_params.items() if value}

  offroad = params.get_bool("IsOffroad")
  custom_torque_being_enabled = _custom_torque_enabled_after_valid_request(params_to_update, compression, attested_keys, offroad)

  saved_any = False
  for key, value in params_to_update.items():
    policy = remote_write_policy(key)
    if policy is None:
      cloudlog.warning(f"sunnylinkd.saveParams.denied: '{key}' is not remotely writable")
      continue

    if policy.get("attestation_required") and key not in attested_keys:
      cloudlog.warning(f"sunnylinkd.saveParams.attestation_required: Missing attestation for '{key}'")
      continue

    # Defense-in-depth: safety-critical torque/control settings are only writable
    # while the device is offroad. They are also subject to capability checks when
    # vehicle information is available (e.g., torque steering must be allowed).
    if key in SAFETY_CRITICAL_REMOTE_GATED_PARAMS:
      if not offroad:
        cloudlog.warning(f"sunnylinkd.saveParams.offroad_required: '{key}' rejected while onroad")
        continue
      if not _torque_settings_allowed():
        cloudlog.warning(f"sunnylinkd.saveParams.capability_rejected: '{key}' rejected, torque not allowed for this vehicle")
        continue

    # Manual override state and values require CustomTorqueParams to be enabled.
    # The dependency can be satisfied by the current value or by an atomic
    # CustomTorqueParams=true write in the same transaction.
    if key in DEPENDS_ON_CUSTOM_TORQUE_PARAMS and not custom_torque_being_enabled:
      cloudlog.warning(f"sunnylinkd.saveParams.dependency_missing: '{key}' requires CustomTorqueParams")
      continue

    decoded_value = value
    if policy:
      try:
        decoded_value = _decode_param_value(value, compression)
      except Exception as e:
        cloudlog.warning(f"sunnylinkd.saveParams.decode_failed: {key} {e}")
        continue

    if not _decoded_value_matches_policy(decoded_value, policy):
      continue

    try:
      save_param_from_base64_encoded_string(key, value, compression)
      saved_any = True
    except Exception as e:
      cloudlog.error(f"sunnylinkd.saveParams.exception {e}")

  if saved_any:
    # Increment version counter for frontend change detection
    try:
      current = int(params.get("ParamsVersion") or "0")
      params.put("ParamsVersion", str(current + 1), block=True)
    except Exception:
      pass


def startLocalProxy(global_end_event: threading.Event, remote_ws_uri: str, local_port: int) -> dict[str, int]:
  sunnylink_dongle_id = params.get("SunnylinkDongleId")
  sunnylink_api = SunnylinkApi(sunnylink_dongle_id)

  cloudlog.debug("athena.startLocalProxy.starting")
  ws = create_connection(
    remote_ws_uri, header={"Authorization": f"Bearer {sunnylink_api.get_token()}"}, enable_multithread=True, sslopt={"cert_reqs": ssl.CERT_NONE}
  )

  return start_local_proxy_shim(global_end_event, local_port, ws)


def main(exit_event: threading.Event | None = None):
  try:
    set_core_affinity([0, 1, 2, 3])
  except Exception:
    cloudlog.exception("failed to set core affinity")

  while sunnylink_need_register(params):
    cloudlog.info("Waiting for sunnylink registration to complete")
    time.sleep(10)

  sunnylink_dongle_id = params.get("SunnylinkDongleId")
  sunnylink_api = SunnylinkApi(sunnylink_dongle_id)
  UploadQueueCache.initialize(upload_queue)

  update_car_list_param()

  ws_uri = f"{SUNNYLINK_ATHENA_HOST}"
  conn_start = None
  conn_retries = 0
  while (exit_event is None or not exit_event.is_set()) and sunnylink_ready(params):
    try:
      if conn_start is None:
        conn_start = time.monotonic()

      cloudlog.event("sunnylinkd.main.connecting_ws", ws_uri=ws_uri, retries=conn_retries)
      ws = create_connection(
        ws_uri,
        header={"Authorization": f"Bearer {sunnylink_api.get_token()}"},
        enable_multithread=True,
        sslopt={"cert_reqs": ssl.CERT_NONE if "localhost" in ws_uri else ssl.CERT_REQUIRED},
        timeout=SUNNYLINK_RECONNECT_TIMEOUT_S,
      )
      cloudlog.event("sunnylinkd.main.connected_ws", ws_uri=ws_uri, retries=conn_retries,
                     duration=time.monotonic() - conn_start)
      conn_start = None

      conn_retries = 0
      cur_upload_items.clear()

      handle_long_poll(ws, exit_event)
    except (KeyboardInterrupt, SystemExit):
      break
    except Exception as e:
      conn_retries += 1
      params.remove("LastSunnylinkPingTime")

      if isinstance(e, (ConnectionError, TimeoutError, WebSocketException)):
        cloudlog.warning(f"sunnylinkd.main.{type(e).__name__}")
      elif isinstance(e, OSError):
        name = errno.errorcode.get(e.errno or -1, "UNKNOWN")
        msg = f"sunnylinkd.main.OSError.{name} ({e.errno})"
        is_expected_error = e.errno in (errno.ENETDOWN, errno.ENETRESET, errno.ENETUNREACH)
        cloudlog.warning(msg) if is_expected_error else cloudlog.exception(msg)
      else:
        cloudlog.exception("sunnylinkd.main.exception")

    time.sleep(backoff(conn_retries))

  if not sunnylink_ready(params):
    cloudlog.debug("Reached end of sunnylinkd.main while sunnylink is not ready. Waiting 60s before retrying")
    time.sleep(60)


if __name__ == "__main__":
  main()
