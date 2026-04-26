## mapd v1.12 Extensions

This directory contains source patches for the external `pfeiferj/mapd` project.
The sunnypilot repository downloads mapd as a prebuilt binary, so these outputs
must be built in a mapd fork before the in-repo consumers can receive data.

`mapd-v1.12-traffic-controls.patch` targets `pfeiferj/mapd` tag `v1.12.0` and:

- preserves OSM node traffic controls while generating offline map files
- stores the first stop sign or traffic signal attached to each way
- emits `MapTrafficControl` and `NextMapTrafficControl` memory params
- filters current-way traffic controls to nodes ahead in the route direction

`mapd-v1.12-map-context.patch` applies after the traffic-control patch and:

- emits current `MapLanes` and `MapRoadContext` memory params
- emits `NextMapLanes` when an upcoming way changes lane count
- keeps lane/context output as telemetry only for sunnypilot consumers

After applying the patches to mapd, regenerate `offline.capnp.go`, rebuild the
release binary, and update the sunnypilot mapd binary URL/version only when that
binary is available.
