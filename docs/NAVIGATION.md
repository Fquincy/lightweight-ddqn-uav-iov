# Paths in this public repository

The main collaborator guide is in `docs/`. The two corrected simulator modules
are in `src/simulator/`. Minimal setup/check commands are in the root `SETUP.md`.
The new lightweight controller/runner remains future work.

| Original private-tree location | Public location |
| --- | --- |
| `Lightweight DDQN/` main handoff documents | `docs/` |
| `Low Complexity/Source/` corrected environment | `src/simulator/` |
| Selected historical model kernels | `src/research_reference/` |
| `Low Complexity/` source and frozen JSON declarations | `reference/low_complexity_source/` |
| Selected Low Complexity reviews | `reference/research_history/` |
| October 31 model validation notes | `reference/modeling/` |
| Original legacy training files | `reference/historical_legacy/` |
| Private alignment reviews | Public technical summary in `docs/LEGACY_ALIGNMENT_SUMMARY.md` |

`provenance/SOURCE_MAP.json` identifies exact original paths and SHA256 values.
Copied historical scripts retain their imports and guards. To inspect a complete
baseline dependency graph, use `reference/low_complexity_source/`, whose adjacent
`Source/` is provided. The selected files in `src/research_reference/` are kernels,
not a new integrated training package. The original compiled backends may still
require historical packages/paths and are not certified portable here.

Full papers are linked by DOI in the reference notes rather than redistributed.
References to an original local PDF describe provenance, not a missing download
promise. New lightweight source should go into its own package and declare a
tested runner before a large run is authorized.
