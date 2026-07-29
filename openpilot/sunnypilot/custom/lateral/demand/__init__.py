"""Lateral demand pipeline (Phase 3) — turns the raw model curvature into the processed
desired curvature the controller tracks.

The three processors (`model_path_processor`, `lane_change_path_shaper`,
`lane_centering_assist`) and the demand contract (`types`) are FAITHFUL PORTS of the legacy
custom-branch modules, relocated verbatim into this namespace (only the cross-module import
path changed). They were already clean, well-structured, deterministic transforms with no
cruft to rewrite — so, exactly as the torque ADR reasons for behavior-carrying code, a
blind clean-room rewrite would risk losing subtle behavior with no engaged-data gate to
catch it. The CLEAN-ARCHITECTURE work here is the composition: `pipeline.LateralDemandPipeline`
replaces the legacy stack/registry/selector with a single ordered pipeline of
individually-toggleable processors and one debug dict.

Validation: the processors keep their deterministic behavior (exercised by behavioral/property
tests). Their per-processor *value* (does each improve real driving) is gated on the engaged
corpus, like the rest of the controller — see
`docs/adr/2026-06-13-clean-room-torque-v2-1-architecture.md` and the restart plan's Phase 3.
"""
