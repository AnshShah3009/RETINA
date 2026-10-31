//! Process-memory probes for the PCD reader, in **their own test binary** on
//! purpose. They read `VmSize` - a process-wide quantity that concurrent test
//! threads inflate (each `run_with_budget` thread alone costs ~70 MB of address
//! space, which is exactly what the warm-up in the first probe below records),
//! so living in a large test binary made the measurement depend on how busy
//! that process happened to be and failed nondeterministically under
//! `cargo test`'s threaded runner. Here each probe is alone in its binary, and
//! the two are additionally serialized with a mutex so both `cargo test` and
//! `nextest` (process per test) see the same determinism.

#[path = "robustness/common/mod.rs"]
mod common;

use common::*;
use std::sync::Mutex;

static PROBE: Mutex<()> = Mutex::new(());

/// The plain-binary path's cap holds, but the *ASCII* path clamps the
/// reservation to 200M points - 2.4 GB - and commits it before looking at the
/// body, so a 90-byte file with `POINTS 40000000000` costs 2.4 GB of memory to
/// discover that it is empty. Not a crash, but a cheap amplification.
#[test]
fn pcd_ascii_hostile_points_reserves_2_4gb_for_an_empty_body() {
    let _probe = PROBE.lock().unwrap();
    let file = pcd_ascii("40000000000", "POINTS 40000000000\n", "");
    // `run_with_budget` runs the parser on a dedicated 4 MiB stack, and the
    // *first* such thread in a process also gets glibc's per-thread malloc
    // arena. Both land in the process's `VmSize`, so measuring before/after
    // across the call attributes ~68 MiB of the harness's own cost to the
    // parser - which is more than this test's 64 MiB budget. Warm the cost up
    // so what is left is the parser's own.
    let _ = run_with_budget(DEFAULT_BUDGET, || Ok(0u8));
    let before = address_space_bytes();
    let cloud = parse_hostile(run_pcd, &file).expect_ok("empty ascii body");
    assert_eq!(cloud.len(), 0, "no data lines means no points");
    let grew = address_space_bytes()
        .zip(before)
        .map(|(after, before)| after.saturating_sub(before))
        .unwrap_or(0);
    assert!(
        grew < 64 * 1024 * 1024,
        "KNOWN BUG: `POINTS 40000000000` on an empty ASCII body grew the address space by \
         {grew} bytes (the reservation is clamped to 200M points, still 2.4 GB); a 90-byte file \
         should not cost gigabytes"
    );
}

/// The address-space probe: whatever the outcome, a hostile header must not
/// commit a 48 GB virtual allocation. (Belt and braces for the byte cap.)
#[test]
fn pcd_hostile_points_does_not_commit_gigabytes() {
    let _probe = PROBE.lock().unwrap();
    let _ = run_with_budget(DEFAULT_BUDGET, || Ok(0u8));
    let before = address_space_bytes();
    let _ = parse_hostile(run_pcd, &pcd_binary("4000000000", "", &[]));
    if let (Some(before), Some(after)) = (before, address_space_bytes()) {
        let grew = after.saturating_sub(before);
        assert!(
            grew < 1024 * 1024 * 1024,
            "address space grew by {} bytes for a hostile POINTS header",
            grew
        );
    }
}
