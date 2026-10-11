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
| `figures/presentation/2026-10-11/` | Figures derived from the recorded trajectories and metrics |
| `results/presentation/2026-10-11/` | Exact result tables, captions, figure index and asset hashes |
| `docs/report/` | Project report in editable Markdown and its source manifest |
| `results/checks/2026-10-11/` | Previously recorded mathematical model checks |
| `docs/reproduction/` | Commands for independent reproduction and inspection |
| `docs/provenance/` | Old-to-new file mapping and integrity records |
| `archive/2026-10-11/source/` | Exact historical source used before organization |
| `archive/2026-10-11/organization/` | Structure notes and path map retained before the README revision |
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

The root README now describes the completed experiments, result tables and project
report. The earlier README remains in the historical source archive. The original
organization manifest records the files and hashes at Step 4.2; later revisions have
their own verification records. The current path map retains each baseline hash and
updates the current README hash without changing its preserved historical source.

The `archive` contains original source files, not a complete executable copy of every
old directory tree. The separate baseline snapshot preserves the complete prior layout.
