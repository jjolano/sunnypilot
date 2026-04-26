## mapd Traffic-Control Extension

This directory contains source patches for the external `pfeiferj/mapd` project.
The sunnypilot repository downloads mapd as a prebuilt binary, so stop-sign and
traffic-signal extraction must be built in a mapd fork before the in-repo
`MapTrafficControl` / `NextMapTrafficControl` consumers can receive data.

`mapd-v1.12-traffic-controls.patch` targets `pfeiferj/mapd` tag `v1.12.0` and:

- preserves OSM node traffic controls while generating offline map files
- stores the first stop sign or traffic signal attached to each way
- emits `MapTrafficControl` and `NextMapTrafficControl` memory params
- filters current-way traffic controls to nodes ahead in the route direction

After applying the patch to mapd, regenerate `offline.capnp.go`, rebuild the
release binary, and update the sunnypilot mapd binary URL/version only when that
binary is available.
