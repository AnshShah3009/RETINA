<!--
Thanks for contributing! Fill in what applies and delete the rest.
CONTRIBUTING.md explains each item: https://github.com/AnshShah3009/RETINA/blob/master/CONTRIBUTING.md
-->

## What and why

<!-- What this changes and why. Link the issue: "Fixes #123". -->

## Kind of change

- [ ] Bug fix
- [ ] Parity fix (now matches OpenCV / Open3D / SciPy / Matplotlib)
- [ ] Performance
- [ ] New feature / API
- [ ] Breaking change (title has `!`, e.g. `feat(imgproc)!: ...`)
- [ ] Docs / tests / CI only

## Evidence

<!--
Bug fix: the test that failed before the fix, and its failure message.
Parity: reference library + version, and how the reference values were produced.
Performance: benchmark before/after (median and spread, same machine) and how
you showed the results are unchanged.
-->

## Checklist

- [ ] `scripts/check.sh` passes (`scripts/check.sh full` if I touched features, GPU code or benches)
- [ ] No compiler warnings (`cargo check --workspace --all-targets`)
- [ ] New behaviour is tested; a bug fix has a test that failed before it
- [ ] No existing test's tolerance was loosened (or the PR explains why the old expectation was wrong)
- [ ] Public items are documented; docs match behaviour
- [ ] `CHANGELOG.md` updated under **Unreleased** (user-visible changes)
- [ ] `docs/bug-log.md` has a row for each defect fixed
- [ ] GPU changes: CPU/GPU parity tests pass; int-div audit, shader count and element-type tables updated

## Notes for reviewers

<!-- Anything you want a second pair of eyes on, untested platforms, follow-ups. -->
