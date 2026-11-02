# Documentation

## Start here

| Document | For |
| --- | --- |
| [../README.md](../README.md) | What RETINA is, quick start, crate overview |
| [ARCHITECTURE.md](ARCHITECTURE.md) | How the crates fit together, the HAL, tensor conventions, where code goes |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | Setting up, what a pull request needs, recipes for common changes |

## Correctness

| Document | Contents |
| --- | --- |
| [PARITY.md](PARITY.md) | Agreement with OpenCV, Open3D, SciPy and Matplotlib: what matches, how it was measured, known differences |
| [bug-log.md](bug-log.md) | Every defect fixed, how it was found, and the evidence for the fix |
| [wgsl_int_div_audit.md](wgsl_int_div_audit.md) | Proof that every integer division in the WGSL shaders has a nonzero divisor |
| [../GAPS.md](../GAPS.md) | Review findings and their status |

## Performance and evaluation

| Document | Contents |
| --- | --- |
| [performance.md](performance.md) | Measured timings and accuracy, with the commands and parameter sweeps behind each default |
| [GPU_BENCHMARK_RESULTS.md](GPU_BENCHMARK_RESULTS.md) | CPU vs GPU benchmark results |
| [evaluation.md](evaluation.md) | Metric definitions, dataset formats, the `cv-bench` CLI, and the reproducibility rules |
| [VS_REFERENCE_COMPARISON.md](VS_REFERENCE_COMPARISON.md) | Capability comparison against visloc-rs |

## Components

| Document | Contents |
| --- | --- |
| [localization.md](localization.md) | The visual localization pipeline and its proven bounds |
| [VIEWER.md](VIEWER.md) | The 3-D viewer: what works, what does not |
| [MIGRATION_FLOAT_GENERICS.md](MIGRATION_FLOAT_GENERICS.md) | Migrating to the float-generic APIs and `half-precision` |

## Design history

[`plans/`](plans/) and [`superpowers/`](superpowers/) hold design notes and
implementation plans written while features were developed. They record intent at
the time and may be out of date; the code and the documents above are
authoritative.
