# Security policy

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Report it privately through GitHub's private vulnerability reporting:
[**Report a vulnerability**](https://github.com/AnshShah3009/RETINA/security/advisories/new).

Include what you can of:

- the affected crate(s), version or commit;
- the input or call sequence that triggers the problem (a file that crashes a
  parser is ideal);
- the impact you observed or expect (crash, hang, memory exhaustion, out-of-bounds
  access, wrong result that could be exploited);
- whether you want to be credited, and how.

You should get an acknowledgement within **7 days**. We will work with you on a
fix and a disclosure date; the default is to publish an advisory together with
the fixed release, and no later than 90 days after the report.

## Supported versions

RETINA is pre-1.0. Security fixes land on `master` and go into the next release;
older releases are not patched.

| Version | Supported |
| --- | --- |
| `master` | Yes |
| latest `0.x` release | Yes |
| older releases | No |

## Scope

In scope, for example:

- **File parsers** in `cv-io` (PLY, PCD, OBJ, STL, LAS, glTF, image and dataset
  loaders) and `cv-videoio`: crashes, hangs, unbounded allocation or out-of-bounds
  reads on malformed input. These are the code most likely to see untrusted data.
- **Network and IPC code** in `cv-distributed`: framing, partial reads,
  shared-memory handling.
- **`unsafe` code** in `cv-hal` (GPU buffer casts) and `cv-distributed` (shared
  memory). The other 27 crates forbid `unsafe`.
- **Python bindings** (`crates/python`): memory-safety issues across the FFI
  boundary.
- GPU kernels that read or write outside their buffers.

Out of scope:

- Wrong numerical results that are not exploitable (please open a normal bug or
  parity issue).
- Vulnerabilities in dependencies without a demonstrated impact on RETINA (report
  them upstream; Dependabot raises dependency updates, and CI runs
  `cargo deny`, which fails on RustSec advisories).
- Denial of service by deliberately huge but well-formed inputs, unless a small
  input causes disproportionate resource use.

## Hardening already in place

- `#![forbid(unsafe_code)]` in 27 of 30 crates.
- Parser robustness tests with deterministic mutation corpora
  (`crates/io/tests/parser_robustness.rs`).
- CI fails on any compiler warning.
- Dependabot keeps Cargo dependencies and GitHub Actions up to date.
