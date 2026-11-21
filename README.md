# Lightweight DDQN for UAV-assisted IoV task offloading

Research toward an implemented lightweight DDQN offloading controller with
measured deployment storage/latency and preserved return/deadline performance.
**This repository currently contains the collaborator plan, corrected simulator
and historical reference implementations. A new lightweight controller and its
training runner have not yet been implemented, frozen or launched.**

## Start here

1. [Collaborator experiment instructions](docs/START_HERE_LIGHTWEIGHT_DDQN_EXPERIMENT_INSTRUCTIONS_2026-11-21.txt)
2. [Reference reading and research gap](docs/REFERENCE_NOTES_2026-11-21.md)
3. [Public navigation and source map](provenance/SOURCE_MAP.json)
4. [Setup and tiny environment check](SETUP.md)
5. [Scope](SCOPE.md) and [publication boundaries](PUBLICATION_SCOPE.md)

The selected recent full-text inspiration is Xu et al., Sensors 2025,
doi:10.3390/s25113403. Hao et al., IEEE TMC 2024,
doi:10.1109/TMC.2024.3350078, is the historical modeling anchor. Reference notes
explain method differences and remaining full-text review tasks. Neither paper
establishes novelty or success of this proposed DDQN extension.

The active simulator is self-contained in `src/simulator/`: vehicle-local,
UAV1/UAV2 and RSU1/RSU2 processing with UAV movement. Cloud and charging are
excluded; UAV energy constraints remain. Security/trust is deferred.
Legacy cloud/charging code is quarantined under `reference/historical_legacy/`
and must not be used as the active experiment. Existing low-complexity source
and model references are provided locally, without a runtime dependency on the
other repository. Some historical runners remain machine-bound; they are not
ready-to-launch commands for a new workstation.

The [low-complexity track](https://github.com/Fquincy/low-complexity-ddqn-uav-iov)
continues independently. Coordinate seeds and preserve its failed/mixed results
and reserved final sets. The first favorable development result is not an
automatic thesis winner: fresh confirmation and an honest cost/policy comparison
remain necessary. A powerful training computer does not itself make a deployed
controller lightweight. Follow the design/engineering/protocol/launch gates.

No source PDFs, personal documents, old checkpoint binaries or virtual
environments are included. Publisher links and provenance identify reference
material; full historical datasets are not hosted in this initial export.
