using Cxx = import "./include/c++.capnp";
$Cxx.namespace("cereal");

@0xb526ba661d550a59;

# custom.capnp: a home for empty structs reserved for custom forks
# These structs are guaranteed to remain reserved and empty in mainline
# cereal, so use these if you want custom events in your fork.

# DO rename the structs
# DON'T change the identifier (e.g. @0x81c2f05a394cf4af)

struct ModularAssistiveDrivingSystem {
  state @0 :ModularAssistiveDrivingSystemState;
  enabled @1 :Bool;
  active @2 :Bool;
  available @3 :Bool;

  enum ModularAssistiveDrivingSystemState {
    disabled @0;
    paused @1;
    enabled @2;
    softDisabling @3;
    overriding @4;
  }
}

struct IntelligentCruiseButtonManagement {
  state @0 :IntelligentCruiseButtonManagementState;
  sendButton @1 :SendButtonState;
  vTarget @2 :Float32;

  enum IntelligentCruiseButtonManagementState {
    inactive @0;      # No button press or default state
    preActive @1;     # Pre-active state before transitioning to increasing or decreasing
    increasing @2;    # Increasing speed
    decreasing @3;    # Decreasing speed
    holding @4;       # Holding steady speed
  }

  enum SendButtonState {
    none @0;
    increase @1;
    decrease @2;
  }
}

# Same struct as Log.RadarState.LeadData
struct LeadData {
  dRel @0 :Float32;
  yRel @1 :Float32;
  vRel @2 :Float32;
  aRel @3 :Float32;
  vLead @4 :Float32;
  dPath @6 :Float32;
  vLat @7 :Float32;
  vLeadK @8 :Float32;
  aLeadK @9 :Float32;
  fcw @10 :Bool;
  status @11 :Bool;
  aLeadTau @12 :Float32;
  modelProb @13 :Float32;
  radar @14 :Bool;
  radarTrackId @15 :Int32 = -1;

  aLeadDEPRECATED @5 :Float32;
}

struct SelfdriveStateSP @0x81c2f05a394cf4af {
  mads @0 :ModularAssistiveDrivingSystem;
  intelligentCruiseButtonManagement @1 :IntelligentCruiseButtonManagement;
  # Engagement-Cycle Latch: the Longitudinal Mode captured by selfdrived when controls
  # engage and held until disengagement. Planner consumers use this, not the Param.
  activeLongitudinalMode @2 :LongitudinalPlanSP.CustomLongitudinal.CustomLongitudinalMode;

  enum AudibleAlert {
    none @0;

    engage @1;
    disengage @2;
    refuse @3;

    warningSoft @4;
    warningImmediate @5;

    prompt @6;
    promptRepeat @7;
    promptDistracted @8;

    # unused, these are reserved for upstream events so we don't collide
    reserved9 @9;
    reserved10 @10;
    reserved11 @11;
    reserved12 @12;
    reserved13 @13;
    reserved14 @14;
    reserved15 @15;
    reserved16 @16;
    reserved17 @17;
    reserved18 @18;
    reserved19 @19;
    reserved20 @20;
    reserved21 @21;
    reserved22 @22;
    reserved23 @23;
    reserved24 @24;
    reserved25 @25;
    reserved26 @26;
    reserved27 @27;
    reserved28 @28;
    reserved29 @29;
    reserved30 @30;

    promptSingleLow @31;
    promptSingleHigh @32;
  }
}

struct ModelManagerSP @0xaedffd8f31e7b55d {
  activeBundle @0 :ModelBundle;
  selectedBundle @1 :ModelBundle;
  availableBundles @2 :List(ModelBundle);

  struct DownloadUri {
    uri @0 :Text;
    sha256 @1 :Text;
  }

  enum DownloadStatus {
    notDownloading @0;
    downloading @1;
    downloaded @2;
    cached @3;
    failed @4;
  }

  struct DownloadProgress {
    status @0 :DownloadStatus;
    progress @1 :Float32;
    eta @2 :UInt32;
  }

  struct Artifact {
    fileName @0 :Text;
    downloadUri @1 :DownloadUri;
    downloadProgress @2 :DownloadProgress;
  }

  struct Model {
    type @0 :Type;
    artifact @1 :Artifact;  # Main artifact
    metadata @2 :Artifact;  # Metadata artifact

    enum Type {
      supercombo @0;
      navigation @1;
      vision @2;
      policy @3;
      offPolicy @4;
      onPolicy @5;
    }
  }

  enum Runner {
    snpe @0;
    tinygrad @1;
    stock @2;
  }

  struct Override {
    key @0 :Text;
    value @1 :Text;
  }

  struct ModelBundle {
    index @0 :UInt32;
    internalName @1 :Text;
    displayName @2 :Text;
    models @3 :List(Model);
    status @4 :DownloadStatus;
    generation @5 :UInt32;
    environment @6 :Text;
    runner @7 :Runner;
    is20hz @8 :Bool;
    ref @9 :Text;
    minimumSelectorVersion @10 :UInt32;
    overrides @11 :List(Override);
  }
}

struct LongitudinalPlanSP @0xf35cc4560bbf6ec2 {
  dec @0 :DynamicExperimentalControl;
  longitudinalPlanSource @1 :LongitudinalPlanSource;
  smartCruiseControl @2 :SmartCruiseControl;
  speedLimit @3 :SpeedLimit;
  vTarget @4 :Float32;
  aTarget @5 :Float32;
  events @6 :List(OnroadEventSP.Event);
  e2eAlerts @7 :E2eAlerts;
  customLongitudinal @8 :CustomLongitudinal;
  longitudinalDebug @9 :LongitudinalDebug;

  struct DynamicExperimentalControl {
    state @0 :DynamicExperimentalControlState;
    enabled @1 :Bool;
    active @2 :Bool;

    enum DynamicExperimentalControlState {
      acc @0;
      blended @1;
    }
  }

  struct CustomLongitudinal {
    mode @0 :CustomLongitudinalMode;
    enabled @1 :Bool;
    active @2 :Bool;
    shouldStop @3 :Bool;
    selectedIntent @4 :Text;
    reason @5 :Text;
    # Stable Fault Class of a Fail-closed custom-longitudinal fault; never raw exception text.
    faultClass @6 :Text;

    enum CustomLongitudinalMode {
      acc @0;
      e2e @1;
      scc @2;
    }
  }

  struct LongitudinalDebug {
    enabled @0 :Bool;
    traceMode @1 :Text;
    vEgo @2 :Float32;
    vCruise @3 :Float32;
    customATarget @4 :Float32;
    customShouldStop @5 :Bool;
    customIntent @6 :Text;
    customReason @7 :Text;
    mpcATarget @8 :Float32;
    mpcShouldStop @9 :Bool;
    modelATarget @10 :Float32;
    modelShouldStop @11 :Bool;
    finalATargetUnclipped @12 :Float32;
    finalATargetClipped @13 :Float32;
    finalShouldStop @14 :Bool;
    accelClipMin @15 :Float32;
    accelClipMax @16 :Float32;
    e2eSource @17 :Bool;

    # Retired/unwritten compatibility slot. Keep @18 so later fields retain their original ordinals.
    struct LeadPathClearance {
      mode @0 :Text;
      effectiveMode @1 :Text;
      applySupported @2 :Bool;
      shadowEligible @3 :Bool;
      blockReason @4 :Text;
      leadIdx @5 :Int32;
      pathYRel @6 :Float32;
      lateralVelocity @7 :Float32;
      tClear @8 :Float32;
      tConflict @9 :Float32;
      confidence @10 :Float32;
      modelProb @11 :Float32;
      ttc @12 :Float32;
      requiredDecel @13 :Float32;
      leadStatus @14 :Bool;
      leadDRel @15 :Float32;
      leadVRel @16 :Float32;
      leadYRel @17 :Float32;
    }

    struct CutInBrakeAssistTrace {
      mode @0 :Text;
      effectiveMode @1 :Text;
      applySupported @2 :Bool;
      eligible @3 :Bool;
      blockReason @4 :Text;
      leadIdx @5 :Int32;
      pathYRel @6 :Float32;
      lateralVelocity @7 :Float32;
      ttc @8 :Float32;
      requiredDecel @9 :Float32;
      proposedCap @10 :Float32;
      confidence @11 :Float32;
    }
    # Retired 2026-07-24: the feature was deleted after the shadow harvest measured it
    # eligible on 0.07% of 101,741 engaged frames. The struct and its @20 ordinal are kept
    # so existing logs still decode and later fields retain their ordinals; nothing writes it.
    struct CurveSpeedConfidenceTrace {
      mode @0 :Text;
      effectiveMode @1 :Text;
      applySupported @2 :Bool;
      eligible @3 :Bool;
      blockReason @4 :Text;
      confidence @5 :Float32;
      proposedCap @6 :Float32;
      source @7 :Text;
      active @8 :Bool;
      currentLatAccel @9 :Float32;
      maxPredLatAccel @10 :Float32;
      preEntryActive @11 :Bool;
    }
    struct StandstillReleaseConfidenceTrace {
      mode @0 :Text;
      effectiveMode @1 :Text;
      applySupported @2 :Bool;
      eligible @3 :Bool;
      blockReason @4 :Text;
      confidence @5 :Float32;
      releaseAllowed @6 :Bool;
      releaseSource @7 :Text;
      releaseReason @8 :Text;
      releaseATarget @9 :Float32;
    }
    struct AccEnvelopeTrace {
      active @0 :Bool;
      wouldCap @1 :Bool;
      capReason @2 :Text;
      allowedATarget @3 :Float32;
      deltaA @4 :Float32;
      desiredGap @5 :Float32;
      timeGap @6 :Float32;
      ttc @7 :Float32;
      usableStoppingGap @8 :Float32;
      requiredStoppingDecel @9 :Float32;
      closingSpeedDecel @10 :Float32;
      jerkLimitedATarget @11 :Float32;
    }

    leadPathClearance @18 :LeadPathClearance;
    cutInBrakeAssist @19 :CutInBrakeAssistTrace;
    curveSpeedConfidence @20 :CurveSpeedConfidenceTrace;
    standstillReleaseConfidence @21 :StandstillReleaseConfidenceTrace;
    accEnvelope @22 :AccEnvelopeTrace;
    struct ScenarioContextTrace {
      mode @0 :Text;
      effectiveMode @1 :Text;
      applySupported @2 :Bool;
      active @3 :Bool;
      scenario @4 :Text;
      confidence @5 :Float32;
      allowedEffect @6 :Text;
      currentEffect @7 :Text;
      roadGrade @8 :Text;
      reason @9 :Text;
      accelCoast @10 :Float32;
      gradeConfidence @11 :Float32;
      estimatedAccelBias @12 :Float32;
      proposedCompensation @13 :Float32;
      blockReason @14 :Text;
    }

    scenarioContextDEPRECATED @23 :ScenarioContextTrace;
    struct DynamicSafetyFloorTrace {
      active @0 :Bool;
      blockReason @1 :Text;
      currentSafeDistance @2 :Float32;
      proposedSafeDistance @3 :Float32;
      deltaSafeDistance @4 :Float32;
      dynamicFloorValue @5 :Float32;
      kinematicFloorViolation @6 :Bool;
      comfortBrakeEffective @7 :Float32;
      latencyS @8 :Float32;
      latAccel @9 :Float32;
      pitch @10 :Float32;
    }
    dynamicSafetyFloor @24 :DynamicSafetyFloorTrace;
    struct MapCoastTrace {
      mode @0 :Text;
      vTarget @1 :Float32;
      distance @2 :Float32;
      eligible @3 :Bool;
      cap @4 :Float32;
      applied @5 :Bool;
      fault @6 :Bool;
      coastDecel @7 :Float32;
    }
    mapCoast @25 :MapCoastTrace;
    struct UphillNetDemandCapTrace {
      mode @0 :Text;
      effectiveMode @1 :Text;
      eligible @2 :Bool;
      wouldCap @3 :Bool;
      applied @4 :Bool;
      blockReason @5 :Text;
      regime @6 :Text;
      source @7 :Text;
      sourceAgeS @8 :Float32;
      carPitch @9 :Float32;
      livePosePitch @10 :Float32;
      pitchZero @11 :Float32;
      relativePitch @12 :Float32;
      filteredGradePercent @13 :Float32;
      profileReady @14 :Bool;
      fitSlope @15 :Float32;
      fitScore @16 :Float32;
      fitPitchSpan @17 :Float32;
      fitResidualMad @18 :Float32;
      fitSampleCount @19 :UInt32;
      ceiling @20 :Float32;
      gradeEnterPercent @21 :Float32;
      gradeExitPercent @22 :Float32;
      gradeAccel @23 :Float32;
      aTargetBefore @24 :Float32;
      aTargetCap @25 :Float32;
      aTargetAfter @26 :Float32;
      requestedNetDemand @27 :Float32;
      deltaA @28 :Float32;
      gradeLoadExceedsCeiling @29 :Bool;
      fitSpeedBandSpread @30 :Float32;
      gradeHeld @31 :Bool;
      researchActuationAllowed @32 :Bool;
      hasLead @33 :Bool;
    }
    uphillNetDemandCap @26 :UphillNetDemandCapTrace;

    # Model-stop telemetry (red light / stop sign diagnosis): the model trajectory's raw
    # predicted rest distance, the anchored/conservative distance the policy planned to
    # (ModelStopAnchor), and whether a model stop is currently committed. 0.0 when none.
    modelStopDistanceRaw @27 :Float32;
    modelStopDistanceUsed @28 :Float32;
    modelStopCommitted @29 :Bool;

    struct DeparturePredictionTrace {
      mode @0 :Text;
      effectiveMode @1 :Text;
      applySupported @2 :Bool;
      phase @3 :Text;
      eligible @4 :Bool;
      blockReason @5 :Text;
      trackId @6 :Int32;
      evidenceS @7 :Float32;
      ageS @8 :Float32;
      predictedGapDelta @9 :Float32;
      wouldCoast @10 :Bool;
      applied @11 :Bool;
      aTargetBefore @12 :Float32;
      aTargetProposed @13 :Float32;
      aTargetAfter @14 :Float32;
      deltaA @15 :Float32;
      researchActuationAllowed @16 :Bool;
    }
    departurePrediction @30 :DeparturePredictionTrace;
    # The pre-smoothing plan target. Since the a_desired plan/command split this — not
    # customATarget — is what reaches the MPC's initial acceleration state, so policy churn
    # propagating into the solver is invisible without it.
    customATargetPlan @31 :Float32;
    struct ConfidenceTrace {
      enum LeadSlot {
        none @0;
        leadOne @1;
        leadTwo @2;
      }
      enum LeadAuthority {
        none @0;
        suppressOnly @1;
        physical @2;
        progressAllowed @3;
      }
      enum CorroborationRefreshSource {
        none @0;
        vision @1;
        radar @2;
      }

      selectedLeadSlot @0 :LeadSlot;
      selectedTrackId @1 :Int32 = -1;
      selectedAuthority @2 :LeadAuthority;
      acquisitionTimerS @3 :Float32;
      flickerGuardTimerS @4 :Float32;
      modelAgeS @5 :Float32 = -1;
      modelStale @6 :Bool;
      modelServiceHealthy @7 :Bool;
      radarServiceHealthy @8 :Bool;
      corroborationHoldRemainingS @9 :Float32;
      corroborationRefreshSource @10 :CorroborationRefreshSource;
      anchorTravelCorroborated @11 :Bool;
      stationaryRadarCorrelationApplied @12 :Bool;
      effectiveCautionFloor @13 :Float32;
      cutOutRemainingS @14 :Float32;
      stopTrust @15 :Float32;
    }
    confidenceTrace @32 :ConfidenceTrace;
  }

  struct SmartCruiseControl {
    vision @0 :Vision;
    map @1 :Map;

    struct Vision {
      state @0 :VisionState;
      vTarget @1 :Float32;
      aTarget @2 :Float32;
      currentLateralAccel @3 :Float32;
      maxPredictedLateralAccel @4 :Float32;
      enabled @5 :Bool;
      active @6 :Bool;
    }

    struct Map {
      state @0 :MapState;
      vTarget @1 :Float32;
      aTarget @2 :Float32;
      enabled @3 :Bool;
      active @4 :Bool;
    }

    enum VisionState {
      disabled @0; # System disabled or inactive.
      enabled @1; # No predicted substantial turn on vision range.
      entering @2; # A substantial turn is predicted ahead, adapting speed to turn comfort levels.
      turning @3; # Actively turning. Managing acceleration to provide a roll on turn feeling.
      leaving @4; # Road ahead straightens. Start to allow positive acceleration.
      overriding @5; # System overriding with manual control.
    }

    enum MapState {
      disabled @0; # System disabled or inactive.
      enabled @1; # No predicted substantial turn on map range.
      turning @2; # Actively turning. Managing acceleration to provide a roll on turn feeling.
      overriding @3; # System overriding with manual control.
    }
  }

  struct SpeedLimit {
    resolver @0 :Resolver;
    assist @1 :Assist;

    struct Resolver {
      speedLimit @0 :Float32;
      distToSpeedLimit @1 :Float32;
      source @2 :Source;
      speedLimitOffset @3 :Float32;
      speedLimitLast @4 :Float32;
      speedLimitFinal @5 :Float32;
      speedLimitFinalLast @6 :Float32;
      speedLimitValid @7 :Bool;
      speedLimitLastValid @8 :Bool;
    }

    struct Assist {
      state @0 :AssistState;
      enabled @1 :Bool;
      active @2 :Bool;
      vTarget @3 :Float32;
      aTarget @4 :Float32;
    }

    enum Source {
      none @0;
      car @1;
      map @2;
    }

    enum AssistState {
      disabled @0;
      inactive @1; # No speed limit set or not enabled by parameter.
      preActive @2;
      pending @3; # Awaiting new speed limit.
      adapting @4; # Reducing speed to match new speed limit.
      active @5; # Cruising at speed limit.
    }
  }

  enum LongitudinalPlanSource {
    cruise @0;
    sccVision @1;
    sccMap @2;
    speedLimitAssist @3;
  }

  struct E2eAlerts {
    greenLightAlert @0 :Bool;
    leadDepartAlert @1 :Bool;
  }
}

struct OnroadEventSP @0xda96579883444c35 {
  events @0 :List(Event);

  struct Event {
    name @0 :EventName;

    # event types
    enable @1 :Bool;
    noEntry @2 :Bool;
    warning @3 :Bool;   # alerts presented only when  enabled or soft disabling
    userDisable @4 :Bool;
    softDisable @5 :Bool;
    immediateDisable @6 :Bool;
    preEnable @7 :Bool;
    permanent @8 :Bool; # alerts presented regardless of openpilot state
    overrideLateral @10 :Bool;
    overrideLongitudinal @9 :Bool;
  }

  enum EventName {
    lkasEnable @0;
    lkasDisable @1;
    manualSteeringRequired @2;
    manualLongitudinalRequired @3;
    silentLkasEnable @4;
    silentLkasDisable @5;
    silentBrakeHold @6;
    silentWrongGear @7;
    silentReverseGear @8;
    silentDoorOpen @9;
    silentSeatbeltNotLatched @10;
    silentParkBrake @11;
    controlsMismatchLateral @12;
    hyundaiRadarTracksConfirmed @13;
    experimentalModeSwitched @14;
    wrongCarModeAlertOnly @15;
    pedalPressedAlertOnly @16;
    laneTurnLeft @17;
    laneTurnRight @18;
    speedLimitPreActive @19;
    speedLimitActive @20;
    speedLimitChanged @21;
    speedLimitPending @22;
    e2eChime @23;
    customLongitudinalFault @24;
  }
}

struct CarParamsSP @0x80ae746ee2596b11 {
  flags @0 :UInt32;        # flags for car specific quirks in sunnypilot
  safetyParam @1 : Int16;  # flags for sunnypilot's custom safety flags
  pcmCruiseSpeed @3 :Bool;
  intelligentCruiseButtonManagementAvailable @4 :Bool;
  enableGasInterceptor @5 :Bool;

  neuralNetworkLateralControl @2 :NeuralNetworkLateralControl;

  struct NeuralNetworkLateralControl {
    model @0 :Model;
    fuzzyFingerprint @1 :Bool;

    struct Model {
      path @0 :Text;
      name @1 :Text;
    }
  }
}

struct CarControlSP @0xa5cd762cd951a455 {
  mads @0 :ModularAssistiveDrivingSystem;
  params @1 :List(Param);
  leadOne @2 :LeadData;
  leadTwo @3 :LeadData;
  intelligentCruiseButtonManagement @4 :IntelligentCruiseButtonManagement;

  struct Param {
    key @0 :Text;
    type @2 :ParamType;
    value @3 :Data;

    valueDEPRECATED @1 :Text; # The data type change may cause issues with backwards compatibility.
  }

  enum ParamType {
    string @0;
    bool @1;
    int @2;
    float @3;
    time @4;
    json @5;
    bytes @6;
  }
}

struct BackupManagerSP @0xf98d843bfd7004a3 {
  backupStatus @0 :Status;
  restoreStatus @1 :Status;
  backupProgress @2 :Float32;
  restoreProgress @3 :Float32;
  lastError @4 :Text;
  currentBackup @5 :BackupInfo;
  backupHistory @6 :List(BackupInfo);

  enum Status {
    idle @0;
    inProgress @1;
    completed @2;
    failed @3;
  }

  struct Version {
    major @0 :UInt16;
    minor @1 :UInt16;
    patch @2 :UInt16;
    build @3 :UInt16;
    branch @4 :Text;
  }

  struct MetadataEntry {
    key @0 :Text;
    value @1 :Text;
    tags @2 :List(Text);
  }

  struct BackupInfo {
    deviceId @0 :Text;
    version @1 :UInt32;
    config @2 :Text;
    isEncrypted @3 :Bool;
    createdAt @4 :Text;  # ISO timestamp
    updatedAt @5 :Text;  # ISO timestamp
    sunnypilotVersion @6 :Version;
    backupMetadata @7 :List(MetadataEntry);
  }
}

struct CarStateSP @0xb86e6369214c01c8 {
  speedLimit @0 :Float32;
}

struct LiveMapDataSP @0xf416ec09499d9d19 {
  speedLimitValid @0 :Bool;
  speedLimit @1 :Float32;
  speedLimitAheadValid @2 :Bool;
  speedLimitAhead @3 :Float32;
  speedLimitAheadDistance @4 :Float32;
  roadName @5 :Text;
}

struct ModelDataV2SP @0xa1680744031fdb2d {
  laneTurnDirection @0 :TurnDirection;

  enum TurnDirection {
    none @0;
    turnLeft @1;
    turnRight @2;
  }
}

struct CustomReserved10 @0xcb9fd56c7057593a {
}

struct CustomReserved11 @0xc2243c65e0340384 {
}

struct CustomReserved12 @0x9ccdc8676701b412 {
}

struct CustomReserved13 @0xcd96dafb67a082d0 {
}

struct CustomReserved14 @0xb057204d7deadf3f {
}

struct CustomReserved15 @0xbd443b539493bc68 {
}

struct CustomReserved16 @0xfc6241ed8877b611 {
}

struct CustomReserved17 @0xa30662f84033036c {
}

struct CustomReserved18 @0xc86a3d38d13eb3ef {
}

struct CustomReserved19 @0xa4f1eb3323f5f582 {
}
