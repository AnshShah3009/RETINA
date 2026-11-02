# Governance

RETINA is a young project with a small maintainer team. This document says who
decides what and how, so contributors know what to expect. It will grow with the
project.

## Roles

**Contributors** — anyone who opens an issue, reviews, or sends a pull request.

**Maintainers** — have write access, review and merge pull requests, triage
issues, cut releases and keep the branches in sync. Current maintainers:

| Maintainer | GitHub | Areas |
| --- | --- | --- |
| Ansh Shah | [AnshShah3009](https://github.com/AnshShah3009) | Project lead; all crates |

A contributor with a track record of good pull requests and reviews in an area
can be invited to become a maintainer for it. Ask in an issue if you are
interested.

## How decisions are made

- **Day-to-day changes** (bug fixes, parity fixes, optimisations, docs) are
  decided in the pull request: one maintainer approval and green CI.
- **API changes and new crates** start as an issue describing the problem, the
  proposed API and how it will be tested. A maintainer agrees on the design before
  substantial code is written.
- **Breaking changes** need to be called out in the PR title (`!` after the type,
  e.g. `feat(imgproc)!: …`), explained in the PR, and listed in
  [CHANGELOG.md](CHANGELOG.md).
- Disagreements are settled with evidence where possible: a test, a benchmark, or
  the reference library's behaviour. Otherwise the project lead decides.

## Review standards

Reviewers check the [contract in CONTRIBUTING.md](CONTRIBUTING.md#the-projects-contract):
a failing-then-passing test for fixes, reference values for parity claims,
before/after numbers and unchanged results for optimisations, no warnings, no
loosened tolerances. Reviewers run the change when the evidence is not already in
CI (for example GPU code tested only on Metal).

## Branches

`master` is the default branch and the target for all pull requests. After a
change merges, a maintainer merges `master` into `develop` and
`research/cubecl` and checks CI on both. See
[CONTRIBUTING.md](CONTRIBUTING.md#branches-and-releases).

## Releases

1. Make sure `master` is green on Linux and macOS CI.
2. Move the **Unreleased** section of [CHANGELOG.md](CHANGELOG.md) under the new
   version and date; bump `version` in the workspace manifests.
3. Tag `vX.Y.Z` on `master` and publish release notes from the changelog.

Versions follow [Semantic Versioning](https://semver.org). Before 1.0, minor
versions may contain breaking changes; the changelog lists them.

## Code of Conduct

Everyone in the project's spaces follows the [Code of Conduct](CODE_OF_CONDUCT.md).
Maintainers enforce it.
