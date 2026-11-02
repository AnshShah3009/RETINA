#!/usr/bin/env bash
# Run the checks CI runs, locally, before you open a pull request.
#
#   scripts/check.sh          quick: fmt, warnings, clippy, workspace tests
#   scripts/check.sh full     quick + doctests, feature builds, examples, benches
#
# Every step runs even if an earlier one fails; the summary at the end lists
# what failed. Exit status is non-zero if anything failed.
#
# GPU tests run when a GPU adapter is present and skip cleanly when it is not.
# CI runs them on the macOS runner (Metal); see CONTRIBUTING.md.

set -u
cd "$(dirname "$0")/.." || exit 1

mode="${1:-quick}"
failed=()

step() {
  local name="$1"
  shift
  printf '\n==> %s\n' "$name"
  if "$@"; then
    printf '    ok: %s\n' "$name"
  else
    printf '    FAILED: %s\n' "$name"
    failed+=("$name")
  fi
}

no_warnings() {
  # `cargo check` exits 0 on warnings; CI fails on them, so this does too.
  local log
  log="$(mktemp)"
  cargo check "$@" --message-format short 2>&1 | tee "$log" >/dev/null
  if grep -E '^[^ ]+:[0-9]+:[0-9]+: (warning|error)' "$log"; then
    rm -f "$log"
    return 1
  fi
  rm -f "$log"
}

tests() {
  # nextest if installed (what CI uses), otherwise cargo test. Both must not
  # stop at the first failing test binary.
  if command -v cargo-nextest >/dev/null 2>&1; then
    cargo nextest run --workspace --no-fail-fast "$@"
  else
    cargo test --workspace --no-fail-fast "$@"
  fi
}

step "rustfmt" cargo fmt --all -- --check
step "no compiler warnings" no_warnings --workspace --all-targets
step "clippy" cargo clippy --workspace --lib -- -W clippy::all
step "workspace tests" tests

if [ "$mode" = "full" ]; then
  step "doctests" cargo test --doc --workspace
  step "half-precision: no warnings" no_warnings -p cv-hal --features half-precision --all-targets
  step "half-precision: hal tests" cargo test -p cv-hal --features half-precision --no-fail-fast
  step "cubecl: no warnings" no_warnings -p cv-hal --features cubecl --all-targets
  step "cubecl: kernel tests" cargo test -p cv-hal --features cubecl --test cubecl_kernels_test
  step "io: las + gltf" cargo test -p cv-io --features las,gltf
  step "examples build" cargo build --workspace --examples
  step "benches build" cargo bench --workspace --no-run
fi

printf '\n'
if [ "${#failed[@]}" -eq 0 ]; then
  printf 'All checks passed (%s).\n' "$mode"
else
  printf 'Failed checks:\n'
  printf '  - %s\n' "${failed[@]}"
  exit 1
fi
