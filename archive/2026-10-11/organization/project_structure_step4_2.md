# Project structure

The source, observations, recorded results and presentation figures have separate locations.
Recorded experiments are grouped by model and completion date.

| Location | Contents |
| --- | --- |
| `src/` | Single-region and two-region model implementations; output-path protection |
| `notebooks/` | Preserved original notebook and the two-region walkthrough |
| `scripts/` | Original reproduction, synthetic experiment runners, verification and notebook/figure builders |
| `tests/` | Existing physical-state, equation and limiting-case checks |
| `data/synthetic/{sird,two_region}/2026-10-11/` | Exact observation CSVs used by the recorded experiments |
| `results/original/2026-10-11/{run_1,run_2}/` | Both original-source reproduction runs |
| `results/{sird,two_region}/2026-10-11/` | Frozen protocols, every final seed, checkpoints, trajectories, baselines and summaries |
| `figures/{original,sird,two_region}/2026-10-11/` | Existing recorded figures, unchanged |
| `results/checks/2026-10-11/` | Previously recorded mathematical model checks |
| `docs/reproduction/` | Commands for independent reproduction and inspection |
| `docs/provenance/` | Old-to-new file mapping and integrity records |
| `archive/2026-10-11/source/` | Exact historical source used before organization |
| `archive/development/` | Pilots and run logs, distinct from final validation runs |
| `runs/` | New reproduction outputs; excluded from version control |

`notebooks/sird-dinn.ipynb` remains byte-identical to the original public GitHub blob.
The numerical contents of both model modules, all recorded observations, protocols,
metrics, checkpoints and existing figures are unchanged. The two-region notebook's
paths are updated for this layout; its earlier version is retained in the source archive.

Historical protocols contain the source hashes and names that existed when those
experiments were run. Those fields are preserved. `path_map.json` identifies the
archived source and current location; historical source hashes are checked against
the archived files, while the unchanged model modules are also checked at their current paths.

The root README is retained pending its separate documentation revision. The project
report and revised figure selection will be prepared in subsequent work.

The `archive` contains original source files, not a complete executable copy of every
old directory tree. The separate baseline snapshot preserves the complete prior layout.
