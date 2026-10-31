//! `LoadCoordinator` implementations must publish and read back the *same*
//! per-device load map: they are two interchangeable ways of coordinating
//! cooperating processes (`create_coordinator` picks between them), so a
//! caller must not have to care which one it got.
//!
//! Regression: `ShmCoordinator` collapsed the map to a single total, stored it
//! in the field `reserve_device` used for compute budgets, and reported the
//! whole sum against `DeviceId(0)`. `update_load({3: 7, 5: 2})` read back as
//! `{0: 9}` - devices 3 and 5 gone, a device 0 entry invented - while
//! `FileCoordinator` returned `{3: 7, 5: 2}`.

use cv_distributed::{
    create_coordinator, CoordinatorType, FileCoordinator, LoadCoordinator, ShmCoordinator,
    SHM_TOTAL_SIZE,
};
use cv_hal::DeviceId;
use std::collections::HashMap;

/// Unique per-test directory/file names (pid + counter) so parallel tests in
/// the same process cannot collide.
fn unique(tag: &str) -> String {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    format!(
        "cv_dist_{}_{}_{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed),
        tag
    )
}

fn sorted(map: &HashMap<DeviceId, usize>) -> Vec<(u32, usize)> {
    let mut v: Vec<(u32, usize)> = map.iter().map(|(k, &v)| (k.0, v)).collect();
    v.sort();
    v
}

/// Entries with a non-zero load, sorted. See the note in
/// `both_coordinators_agree_on_the_published_per_device_load` about explicit
/// zero entries.
fn nonzero(map: &HashMap<DeviceId, usize>) -> Vec<(u32, usize)> {
    let mut v: Vec<(u32, usize)> = map
        .iter()
        .filter(|(_, &l)| l > 0)
        .map(|(k, &l)| (k.0, l))
        .collect();
    v.sort();
    v
}

fn load(pairs: &[(u32, usize)]) -> HashMap<DeviceId, usize> {
    pairs.iter().map(|&(d, l)| (DeviceId(d), l)).collect()
}

struct FileFixture {
    coord: FileCoordinator,
    dir: std::path::PathBuf,
}

impl FileFixture {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(unique(tag));
        let coord = FileCoordinator::new(dir.clone());
        Self { coord, dir }
    }
}

impl Drop for FileFixture {
    fn drop(&mut self) {
        self.coord.cleanup();
        // Remove only the file this process owns, then the (empty) directory.
        let _ = std::fs::remove_file(self.dir.join(format!("{}.load", std::process::id())));
        let _ = std::fs::remove_file(self.dir.join(format!("{}.tmp", std::process::id())));
        let _ = std::fs::remove_dir(&self.dir);
    }
}

#[test]
fn both_coordinators_agree_on_the_published_per_device_load() {
    let cases: &[&[(u32, usize)]] = &[
        &[],
        &[(0, 3)],
        &[(3, 7), (5, 2)],
        &[(1, 0), (2, 9)],
        &[(0, 1), (1, 1), (2, 1), (15, 4)],
    ];

    let fixture = FileFixture::new("parity");
    let shm = ShmCoordinator::new(&unique("parity"), SHM_TOTAL_SIZE).unwrap();

    for case in cases {
        let published = load(case);

        fixture.coord.update_load(&published).unwrap();
        let from_file = fixture.coord.get_global_load().unwrap();

        shm.update_load(&published).unwrap();
        let from_shm = shm.get_global_load().unwrap();

        // A published zero and an absent device state the same thing ("no load
        // here"): the scheduler reads the aggregate with
        // `.get(&device).copied().unwrap_or(0)`. `FileCoordinator` round-trips
        // an explicit zero because its on-disk format has one line per
        // published device, while the shm slot has no bit for "published,
        // explicitly 0". Everything else must match exactly.
        assert_eq!(
            nonzero(&from_file),
            nonzero(&from_shm),
            "LoadCoordinator implementations disagree for {case:?}"
        );

        // Control: what they agree on is exactly the load that was published -
        // not a total parked on device 0, and not a count of processes.
        let expected: Vec<(u32, usize)> = {
            let mut v: Vec<(u32, usize)> = case.iter().copied().filter(|&(_, l)| l > 0).collect();
            v.sort();
            v
        };
        assert_eq!(nonzero(&from_shm), expected, "shm load != published load");
        assert_eq!(nonzero(&from_file), expected, "file load != published load");
    }
}

#[test]
fn publishing_a_new_load_replaces_the_previous_one() {
    let shm = ShmCoordinator::new(&unique("replace"), SHM_TOTAL_SIZE).unwrap();
    let fixture = FileFixture::new("replace");

    fixture.coord.update_load(&load(&[(3, 7), (5, 2)])).unwrap();
    shm.update_load(&load(&[(3, 7), (5, 2)])).unwrap();
    assert_eq!(
        sorted(&shm.get_global_load().unwrap()),
        vec![(3, 7), (5, 2)]
    );

    // A process that becomes idle publishes an empty map; the stale entries
    // must disappear rather than accumulate.
    fixture.coord.update_load(&load(&[])).unwrap();
    shm.update_load(&load(&[])).unwrap();
    assert!(
        shm.get_global_load().unwrap().is_empty(),
        "idle process kept reporting its previous load"
    );
    assert!(fixture.coord.get_global_load().unwrap().is_empty());

    // Control: a later non-empty publish is visible again.
    fixture.coord.update_load(&load(&[(5, 1)])).unwrap();
    shm.update_load(&load(&[(5, 1)])).unwrap();
    assert_eq!(sorted(&shm.get_global_load().unwrap()), vec![(5, 1)]);
}

#[test]
fn create_coordinator_returns_a_working_coordinator_for_both_kinds() {
    let dir = std::env::temp_dir().join(unique("factory_file"));
    let file = create_coordinator(CoordinatorType::File { path: dir.clone() }).unwrap();
    file.update_load(&load(&[(2, 4)])).unwrap();
    assert_eq!(sorted(&file.get_global_load().unwrap()), vec![(2, 4)]);
    file.cleanup();
    let _ = std::fs::remove_file(dir.join(format!("{}.load", std::process::id())));
    let _ = std::fs::remove_dir(&dir);

    let shm = create_coordinator(CoordinatorType::SharedMemory {
        name: unique("factory_shm"),
        size: SHM_TOTAL_SIZE,
    })
    .unwrap();
    shm.update_load(&load(&[(2, 4)])).unwrap();
    assert_eq!(sorted(&shm.get_global_load().unwrap()), vec![(2, 4)]);
}
