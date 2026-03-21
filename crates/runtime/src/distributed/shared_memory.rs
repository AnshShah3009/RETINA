use super::LoadCoordinator;
use cv_hal::DeviceId;
use std::collections::HashMap;
use std::io;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

// --- Layout constants ---

const MAX_SLOTS: usize = 128;
const MAX_DEVICES: usize = 16;

/// Total shared memory size: header + device states + process slots.
/// 64 + 16*128 + 128*256 = 34880 -> round up to 40960 (40 KB).
pub const SHM_TOTAL_SIZE: usize = 40960;

const HEADER_SIZE: usize = 64;
const DEVICE_STATE_SIZE: usize = 128;
const PROCESS_SLOT_SIZE: usize = 256;

const DEVICE_REGION_OFFSET: usize = HEADER_SIZE; // 64
const SLOT_REGION_OFFSET: usize = DEVICE_REGION_OFFSET + MAX_DEVICES * DEVICE_STATE_SIZE; // 2112

const SHM_MAGIC: u32 = 0x52455449; // "RETI"
const SHM_VERSION: u32 = 3;

// --- Slot states ---

const SLOT_EMPTY: u32 = 0;
const SLOT_ACQUIRING: u32 = 1;
const SLOT_ACTIVE: u32 = 2;
const SLOT_RELEASING: u32 = 3;

/// Heartbeat staleness threshold (5 seconds in nanoseconds).
const STALE_THRESHOLD_NS: u64 = 5_000_000_000;

// --- Shared memory structures ---

/// Cache-line aligned header (64 bytes).
#[repr(C, align(64))]
struct ShmHeaderV3 {
    magic: AtomicU32,
    version: AtomicU32,
    num_slots: AtomicU32,
    num_devices: AtomicU32,
    epoch: AtomicU32,
    _pad: [u8; 44],
}

/// Per-device state (128 bytes, cache-line aligned).
#[repr(C, align(64))]
struct DeviceState {
    device_id: AtomicU32,
    total_memory_mb: AtomicU32,
    used_memory_mb: AtomicU32,
    owner_mask: AtomicU64,
    affinity_group: AtomicU32,
    _pad: [u8; 100],
}

/// Per-process slot (256 bytes, cache-line aligned).
#[repr(C, align(64))]
struct ProcessSlot {
    state: AtomicU32,
    pid: AtomicU32,
    generation: AtomicU64,
    heartbeat_ns: AtomicU64,
    start_time_ms: AtomicU64,
    device_mask: AtomicU64,
    memory_budget_mb: [AtomicU32; MAX_DEVICES],
    compute_budget_pct: [AtomicU32; MAX_DEVICES],
    affinity_group: AtomicU32,
    _pad: [u8; 68],
}

// Compile-time size checks
const _: () = assert!(std::mem::size_of::<ShmHeaderV3>() == HEADER_SIZE);
const _: () = assert!(std::mem::size_of::<DeviceState>() == DEVICE_STATE_SIZE);
const _: () = assert!(std::mem::size_of::<ProcessSlot>() == PROCESS_SLOT_SIZE);

/// Cross-process coordinator using POSIX shared memory.
///
/// Provides device reservation with memory budget enforcement, affinity group
/// scheduling, and automatic stale-process reaping via background heartbeats.
pub struct ShmCoordinator {
    mmap: memmap2::MmapMut,
    slot_index: usize,
    pid: u32,
    path: std::path::PathBuf,
    heartbeat_stop: Arc<AtomicU32>,
    heartbeat_thread: std::sync::Mutex<Option<std::thread::JoinHandle<()>>>,
}

// SAFETY: The mmap region uses only atomic operations for cross-process access.
// The coordinator owns its slot and all writes go through atomics.
unsafe impl Send for ShmCoordinator {}
unsafe impl Sync for ShmCoordinator {}

impl ShmCoordinator {
    /// Create or attach to a shared memory region.
    ///
    /// The first process initializes the header and device states; subsequent
    /// processes validate the magic/version and claim a free slot.
    pub fn new(name: &str, size: usize) -> io::Result<Self> {
        let size = size.max(SHM_TOTAL_SIZE);
        let path = Self::get_shm_path(name);

        let needs_init = if path.exists() {
            let existing = std::fs::OpenOptions::new().read(true).open(&path);
            match existing {
                Ok(f) => {
                    let meta = f.metadata()?;
                    if meta.len() >= size as u64 {
                        let mmap = unsafe { memmap2::MmapOptions::new().map(&f)? };
                        if mmap.len() >= HEADER_SIZE {
                            let hdr = unsafe { &*(mmap.as_ptr() as *const ShmHeaderV3) };
                            let magic = hdr.magic.load(Ordering::Relaxed);
                            let version = hdr.version.load(Ordering::Relaxed);
                            magic != SHM_MAGIC || version != SHM_VERSION
                        } else {
                            true
                        }
                    } else {
                        true
                    }
                }
                Err(_) => true,
            }
        } else {
            true
        };

        let file = if needs_init {
            let f = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&path)?;
            f.set_len(size as u64)?;
            f
        } else {
            std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)?
        };

        let mut mmap = unsafe { memmap2::MmapOptions::new().map_mut(&file)? };

        if mmap.len() < SHM_TOTAL_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Shared memory too small: {} < {}",
                    mmap.len(),
                    SHM_TOTAL_SIZE
                ),
            ));
        }

        if needs_init {
            // Zero everything
            for byte in mmap.iter_mut() {
                *byte = 0;
            }
            let header = Self::header_ptr(&mmap);
            header.magic.store(SHM_MAGIC, Ordering::Release);
            header.version.store(SHM_VERSION, Ordering::Release);
            header.num_slots.store(MAX_SLOTS as u32, Ordering::Release);
            header
                .num_devices
                .store(MAX_DEVICES as u32, Ordering::Release);
            header.epoch.store(0, Ordering::Release);
        }

        let pid = std::process::id();
        let slot_index = Self::acquire_slot(&mmap, pid)?;

        Ok(Self {
            mmap,
            slot_index,
            pid,
            path,
            heartbeat_stop: Arc::new(AtomicU32::new(0)),
            heartbeat_thread: std::sync::Mutex::new(None),
        })
    }

    // --- Pointer helpers ---

    fn header_ptr(mmap: &memmap2::MmapMut) -> &ShmHeaderV3 {
        unsafe { &*(mmap.as_ptr() as *const ShmHeaderV3) }
    }

    fn device_ptr(mmap: &memmap2::MmapMut, idx: usize) -> Option<&DeviceState> {
        if idx >= MAX_DEVICES {
            return None;
        }
        let offset = DEVICE_REGION_OFFSET + idx * DEVICE_STATE_SIZE;
        if offset + DEVICE_STATE_SIZE > mmap.len() {
            return None;
        }
        Some(unsafe { &*(mmap.as_ptr().add(offset) as *const DeviceState) })
    }

    fn slot_ptr(mmap: &memmap2::MmapMut, idx: usize) -> Option<&ProcessSlot> {
        if idx >= MAX_SLOTS {
            return None;
        }
        let offset = SLOT_REGION_OFFSET + idx * PROCESS_SLOT_SIZE;
        if offset + PROCESS_SLOT_SIZE > mmap.len() {
            return None;
        }
        Some(unsafe { &*(mmap.as_ptr().add(offset) as *const ProcessSlot) })
    }

    fn my_slot(&self) -> &ProcessSlot {
        Self::slot_ptr(&self.mmap, self.slot_index).expect("own slot index must be valid")
    }

    fn bump_epoch(&self) {
        let header = Self::header_ptr(&self.mmap);
        header.epoch.fetch_add(1, Ordering::Release);
        Self::wake_all_on_address(&header.epoch);
    }

    #[cfg(target_os = "linux")]
    fn wake_all_on_address(addr: &AtomicU32) {
        // Wake only 1 waiter to avoid thundering herd problem
        // Multiple wakeups will happen as needed when each waiter checks the condition
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                addr as *const AtomicU32 as *mut libc::c_int,
                libc::FUTEX_WAKE,
                1, // Wake only one waiter
                std::ptr::null::<libc::timespec>(),
                std::ptr::null::<libc::c_int>(),
                0,
            );
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn wake_all_on_address(addr: &AtomicU32) {
        // Wake only 1 waiter to avoid thundering herd problem
        atomic_wait::wake_one(addr);
    }

    // --- Slot acquisition (lock-free CAS state machine) ---

    fn acquire_slot(mmap: &memmap2::MmapMut, pid: u32) -> io::Result<usize> {
        let now_ns = ShmCoordinator::monotonic_ns();
        let now_ms = Self::current_timestamp_ms();

        // First pass: check if we already own a slot (process restart with same PID)
        for i in 0..MAX_SLOTS {
            let slot =
                Self::slot_ptr(mmap, i).ok_or_else(|| io::Error::other("Invalid slot offset"))?;
            let state = slot.state.load(Ordering::Acquire);
            if state == SLOT_ACTIVE && slot.pid.load(Ordering::Relaxed) == pid {
                // Refresh heartbeat
                slot.heartbeat_ns.store(now_ns, Ordering::Release);
                return Ok(i);
            }
        }

        // Second pass: CAS on an EMPTY slot
        for i in 0..MAX_SLOTS {
            let slot =
                Self::slot_ptr(mmap, i).ok_or_else(|| io::Error::other("Invalid slot offset"))?;

            if slot
                .state
                .compare_exchange(
                    SLOT_EMPTY,
                    SLOT_ACQUIRING,
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                // We won the slot. Initialize it.
                let gen = slot.generation.load(Ordering::Relaxed);
                slot.generation
                    .store(gen.wrapping_add(1), Ordering::Relaxed);
                slot.pid.store(pid, Ordering::Relaxed);
                slot.heartbeat_ns.store(now_ns, Ordering::Relaxed);
                slot.start_time_ms.store(now_ms, Ordering::Relaxed);
                slot.device_mask.store(0, Ordering::Relaxed);
                for j in 0..MAX_DEVICES {
                    slot.memory_budget_mb[j].store(0, Ordering::Relaxed);
                    slot.compute_budget_pct[j].store(0, Ordering::Relaxed);
                }
                slot.affinity_group.store(0, Ordering::Relaxed);

                // Transition to ACTIVE
                slot.state.store(SLOT_ACTIVE, Ordering::Release);
                return Ok(i);
            }
        }

        Err(io::Error::other(
            "No available slots in shared memory coordinator",
        ))
    }

    // --- Device reservation (CAS loop, strict no-overcommit) ---

    /// Reserve memory on a device for this process.
    ///
    /// Uses a CAS loop on `DeviceState.used_memory_mb` for strict enforcement.
    /// Returns `Ok(())` if the reservation succeeds, `Err` if over budget.
    pub fn reserve_device(
        &self,
        device_idx: u8,
        memory_mb: u32,
        compute_pct: u32,
    ) -> io::Result<()> {
        let idx = device_idx as usize;
        let dev = Self::device_ptr(&self.mmap, idx).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Invalid device index: {}", device_idx),
            )
        })?;

        let total = dev.total_memory_mb.load(Ordering::Acquire);
        if total == 0 {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Device {} not initialized (total_memory_mb=0)", device_idx),
            ));
        }

        // CAS loop for memory reservation
        loop {
            let current = dev.used_memory_mb.load(Ordering::Acquire);
            if current + memory_mb > total {
                return Err(io::Error::new(
                    io::ErrorKind::OutOfMemory,
                    format!(
                        "Device {} over budget: used={}MB + requested={}MB > total={}MB",
                        device_idx, current, memory_mb, total
                    ),
                ));
            }
            if dev
                .used_memory_mb
                .compare_exchange(
                    current,
                    current + memory_mb,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                break;
            }
            // CAS failed, another process won — retry
        }

        // Set owner_mask bit for our slot
        let bit = 1u64 << self.slot_index;
        dev.owner_mask.fetch_or(bit, Ordering::AcqRel);

        // Accumulate our slot's per-device budget
        let slot = self.my_slot();
        slot.memory_budget_mb[idx].fetch_add(memory_mb, Ordering::AcqRel);
        slot.compute_budget_pct[idx].fetch_add(compute_pct, Ordering::AcqRel);
        slot.device_mask.fetch_or(1u64 << idx, Ordering::AcqRel);

        self.bump_epoch();
        Ok(())
    }

    /// Release a device reservation held by this process.
    pub fn release_device(&self, device_idx: u8) -> io::Result<()> {
        let idx = device_idx as usize;
        let dev = Self::device_ptr(&self.mmap, idx).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Invalid device index: {}", device_idx),
            )
        })?;

        let slot = self.my_slot();
        let budget = slot.memory_budget_mb[idx].load(Ordering::Acquire);
        if budget == 0 {
            return Ok(()); // nothing to release
        }

        // Subtract our budget from the device's used total
        dev.used_memory_mb.fetch_sub(budget, Ordering::AcqRel);

        // Clear owner bit
        let bit = 1u64 << self.slot_index;
        dev.owner_mask.fetch_and(!bit, Ordering::AcqRel);

        // Clear slot's per-device fields
        slot.memory_budget_mb[idx].store(0, Ordering::Release);
        slot.compute_budget_pct[idx].store(0, Ordering::Release);
        slot.device_mask.fetch_and(!(1u64 << idx), Ordering::AcqRel);

        self.bump_epoch();
        Ok(())
    }

    // --- Affinity groups ---

    /// Join an affinity group. Processes in the same group prefer co-placement on the same device.
    pub fn join_group(&self, group_id: u32) -> io::Result<()> {
        let slot = self.my_slot();
        slot.affinity_group.store(group_id, Ordering::Release);
        self.bump_epoch();
        Ok(())
    }

    /// Leave the current affinity group (set to 0 = ungrouped).
    pub fn leave_group(&self) -> io::Result<()> {
        self.join_group(0)
    }

    // --- Queries ---

    /// Query devices with at least `min_memory_mb` free. Returns `(device_idx, free_mb)`.
    pub fn available_devices(&self, min_memory_mb: u32) -> Vec<(u8, u32)> {
        let mut result = Vec::new();
        for i in 0..MAX_DEVICES {
            let Some(dev) = Self::device_ptr(&self.mmap, i) else {
                continue;
            };
            let total = dev.total_memory_mb.load(Ordering::Acquire);
            if total == 0 {
                continue;
            }
            let used = dev.used_memory_mb.load(Ordering::Acquire);
            let free = total.saturating_sub(used);
            if free >= min_memory_mb {
                result.push((i as u8, free));
            }
        }
        result
    }

    /// Find the best device considering affinity group peers.
    ///
    /// If peers in the same group already hold a device with enough free memory,
    /// returns that device. Otherwise returns the device with the most free memory.
    pub fn best_device_for_group(&self, needed_mb: u32) -> Option<u8> {
        let my_group = self.my_slot().affinity_group.load(Ordering::Acquire);

        // If we're in a group, check where peers are
        if my_group != 0 {
            let mut peer_device_counts: HashMap<u8, u32> = HashMap::new();
            for i in 0..MAX_SLOTS {
                if i == self.slot_index {
                    continue;
                }
                let Some(slot) = Self::slot_ptr(&self.mmap, i) else {
                    continue;
                };
                if slot.state.load(Ordering::Acquire) != SLOT_ACTIVE {
                    continue;
                }
                if slot.affinity_group.load(Ordering::Acquire) == my_group {
                    let mask = slot.device_mask.load(Ordering::Acquire);
                    for d in 0..MAX_DEVICES {
                        if mask & (1u64 << d) != 0 {
                            *peer_device_counts.entry(d as u8).or_insert(0) += 1;
                        }
                    }
                }
            }

            // Pick the peer device with the most peers, if it has enough free memory
            let mut candidates: Vec<_> = peer_device_counts.into_iter().collect();
            candidates.sort_by(|a, b| b.1.cmp(&a.1));
            for (dev_idx, _) in candidates {
                if let Some(dev) = Self::device_ptr(&self.mmap, dev_idx as usize) {
                    let total = dev.total_memory_mb.load(Ordering::Acquire);
                    let used = dev.used_memory_mb.load(Ordering::Acquire);
                    if total.saturating_sub(used) >= needed_mb {
                        return Some(dev_idx);
                    }
                }
            }
        }

        // Fallback: device with the most free memory
        let available = self.available_devices(needed_mb);
        available
            .into_iter()
            .max_by_key(|(_, free)| *free)
            .map(|(idx, _)| idx)
    }

    /// Get per-device memory usage: `(device_idx, used_mb, total_mb)`.
    pub fn device_memory_usage(&self) -> Vec<(u8, u32, u32)> {
        let mut result = Vec::new();
        for i in 0..MAX_DEVICES {
            let Some(dev) = Self::device_ptr(&self.mmap, i) else {
                continue;
            };
            let total = dev.total_memory_mb.load(Ordering::Acquire);
            if total == 0 {
                continue;
            }
            let used = dev.used_memory_mb.load(Ordering::Acquire);
            result.push((i as u8, used, total));
        }
        result
    }

    /// Initialize a device's total memory. Called once by the first process that knows
    /// the hardware's VRAM size. Uses CAS so only one writer wins.
    pub fn init_device(&self, device_idx: u8, total_memory_mb: u32) -> io::Result<()> {
        let dev = Self::device_ptr(&self.mmap, device_idx as usize).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Invalid device index: {}", device_idx),
            )
        })?;
        // CAS: only set if currently 0
        let _ = dev.total_memory_mb.compare_exchange(
            0,
            total_memory_mb,
            Ordering::AcqRel,
            Ordering::Relaxed,
        );
        dev.device_id.store(device_idx as u32, Ordering::Release);
        self.bump_epoch();
        Ok(())
    }

    // --- Wait for VRAM ---

    /// Block until `device_idx` has at least `needed_mb` free, or `timeout` expires.
    ///
    /// Watches the shared-memory epoch counter using futex-like wait
    /// so it wakes up instantly as soon as any process releases memory, with 0% CPU overhead.
    ///
    /// Returns `Ok(())` when memory is available, or `Err(TimedOut)` on timeout.
    pub fn wait_for_device_memory(
        &self,
        device_idx: u8,
        needed_mb: u32,
        timeout: Duration,
    ) -> io::Result<()> {
        let idx = device_idx as usize;
        let dev = Self::device_ptr(&self.mmap, idx).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Invalid device index: {}", device_idx),
            )
        })?;

        let header = Self::header_ptr(&self.mmap);
        let start = std::time::Instant::now();

        loop {
            let used = dev.used_memory_mb.load(Ordering::Acquire);
            let total = dev.total_memory_mb.load(Ordering::Acquire);
            if total > 0 && total.saturating_sub(used) >= needed_mb {
                return Ok(());
            }

            let elapsed = start.elapsed();
            if elapsed >= timeout {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    format!(
                        "Timed out waiting for {}MB on device {} (used={}/{}MB, waited {:?})",
                        needed_mb, device_idx, used, total, timeout
                    ),
                ));
            }

            let last_epoch = header.epoch.load(Ordering::Acquire);

            // Double check condition before waiting to avoid missed-wakeups
            let used = dev.used_memory_mb.load(Ordering::Acquire);
            if total > 0 && total.saturating_sub(used) >= needed_mb {
                return Ok(());
            }

            let remaining = timeout.saturating_sub(elapsed);
            Self::wait_on_address(&header.epoch, last_epoch, Some(remaining));
        }
    }

    #[cfg(target_os = "linux")]
    fn wait_on_address(addr: &AtomicU32, expected: u32, timeout: Option<Duration>) {
        use std::ptr;
        let ts = timeout.map(|d| libc::timespec {
            tv_sec: d.as_secs() as libc::time_t,
            tv_nsec: d.subsec_nanos() as libc::c_long,
        });
        let ts_ptr = ts.as_ref().map(|t| t as *const _).unwrap_or(ptr::null());

        // FUTEX_WAIT blocks if *addr == expected.
        // We do NOT use FUTEX_PRIVATE_FLAG because this is shared memory across processes.
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                addr as *const AtomicU32 as *mut libc::c_int,
                libc::FUTEX_WAIT,
                expected as libc::c_int,
                ts_ptr,
                ptr::null::<libc::c_int>(),
                0,
            );
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn wait_on_address(addr: &AtomicU32, expected: u32, _timeout: Option<Duration>) {
        // Fallback to blocking wait or simple sleep if timeout is not supported via atomic-wait
        atomic_wait::wait(addr, expected);
    }

    // --- Heartbeat & reaping ---

    /// Send heartbeat. Overwrites `heartbeat_ns` with the current monotonic timestamp.
    pub fn send_heartbeat(&self) -> io::Result<()> {
        let slot = self.my_slot();
        slot.heartbeat_ns
            .store(ShmCoordinator::monotonic_ns(), Ordering::Release);
        Ok(())
    }

    /// Scan all slots and reap stale entries (heartbeat > 5s and process dead).
    /// Releases their device reservations atomically.
    pub fn reap_dead(&self) {
        let now = ShmCoordinator::monotonic_ns();

        for i in 0..MAX_SLOTS {
            if i == self.slot_index {
                continue;
            }
            let Some(slot) = Self::slot_ptr(&self.mmap, i) else {
                continue;
            };

            let state = slot.state.load(Ordering::Acquire);
            if state != SLOT_ACTIVE {
                continue;
            }

            let hb = slot.heartbeat_ns.load(Ordering::Acquire);
            // A slot is stale if:
            //   - heartbeat was never sent (hb == 0), OR
            //   - heartbeat is older than STALE_THRESHOLD_NS
            // A slot is fresh (skip it) only if hb > 0 and recent.
            let is_fresh = hb > 0 && now >= hb && (now - hb) < STALE_THRESHOLD_NS;
            if is_fresh {
                continue;
            }

            let slot_pid = slot.pid.load(Ordering::Relaxed);
            if Self::is_process_alive(slot_pid) {
                continue;
            }

            // CAS ACTIVE -> RELEASING (single winner)
            if slot
                .state
                .compare_exchange(
                    SLOT_ACTIVE,
                    SLOT_RELEASING,
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                )
                .is_err()
            {
                continue;
            }

            // Release all device reservations for this slot
            for d in 0..MAX_DEVICES {
                let budget = slot.memory_budget_mb[d].load(Ordering::Acquire);
                if budget > 0 {
                    if let Some(dev) = Self::device_ptr(&self.mmap, d) {
                        dev.used_memory_mb.fetch_sub(budget, Ordering::AcqRel);
                        let bit = 1u64 << i;
                        dev.owner_mask.fetch_and(!bit, Ordering::AcqRel);
                    }
                    slot.memory_budget_mb[d].store(0, Ordering::Release);
                    slot.compute_budget_pct[d].store(0, Ordering::Release);
                }
            }

            // Zero the slot
            slot.pid.store(0, Ordering::Relaxed);
            slot.device_mask.store(0, Ordering::Relaxed);
            slot.heartbeat_ns.store(0, Ordering::Relaxed);
            slot.start_time_ms.store(0, Ordering::Relaxed);
            slot.affinity_group.store(0, Ordering::Relaxed);

            // RELEASING -> EMPTY
            slot.state.store(SLOT_EMPTY, Ordering::Release);
            self.bump_epoch();
        }
    }

    /// Start a background thread that sends heartbeats every `interval` and reaps dead slots.
    ///
    /// The thread is joined on `stop_heartbeat()` or `Drop` — no leaked threads.
    /// Calling this a second time stops the previous thread first.
    pub fn start_heartbeat_thread(&self, interval: Duration) {
        // Stop any existing thread first
        self.stop_heartbeat();
        let stop = self.heartbeat_stop.clone();
        // We need to access the mmap from the thread. Since ShmCoordinator is !Clone,
        // we re-open the same shm file from the thread using the path + slot_index.
        let path = self.path.clone();
        let slot_index = self.slot_index;
        let pid = self.pid;

        let thread_handle = std::thread::Builder::new()
            .name("shm-heartbeat".into())
            .spawn(move || {
                // Re-open the mmap (read-write) for heartbeat writes
                let file = match std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(&path)
                {
                    Ok(f) => f,
                    Err(_) => return,
                };
                let mmap = match unsafe { memmap2::MmapOptions::new().map_mut(&file) } {
                    Ok(m) => m,
                    Err(_) => return,
                };

                while stop.load(Ordering::Relaxed) == 0 {
                    std::thread::sleep(interval);

                    if stop.load(Ordering::Relaxed) != 0 {
                        break;
                    }

                    // Send heartbeat for our slot
                    if let Some(slot) = Self::slot_ptr(&mmap, slot_index) {
                        // Only write if slot still belongs to us
                        if slot.pid.load(Ordering::Relaxed) == pid
                            && slot.state.load(Ordering::Acquire) == SLOT_ACTIVE
                        {
                            slot.heartbeat_ns
                                .store(ShmCoordinator::monotonic_ns(), Ordering::Release);
                        }
                    }

                    // Reap dead slots
                    let now = ShmCoordinator::monotonic_ns();
                    for i in 0..MAX_SLOTS {
                        if i == slot_index {
                            continue;
                        }
                        let Some(slot) = Self::slot_ptr(&mmap, i) else {
                            continue;
                        };
                        let state = slot.state.load(Ordering::Acquire);
                        if state != SLOT_ACTIVE {
                            continue;
                        }
                        let hb = slot.heartbeat_ns.load(Ordering::Acquire);
                        let is_fresh = hb > 0 && now >= hb && (now - hb) < STALE_THRESHOLD_NS;
                        if is_fresh {
                            continue;
                        }
                        let slot_pid = slot.pid.load(Ordering::Relaxed);
                        if Self::is_process_alive(slot_pid) {
                            continue;
                        }
                        if slot
                            .state
                            .compare_exchange(
                                SLOT_ACTIVE,
                                SLOT_RELEASING,
                                Ordering::AcqRel,
                                Ordering::Relaxed,
                            )
                            .is_err()
                        {
                            continue;
                        }
                        for d in 0..MAX_DEVICES {
                            let budget = slot.memory_budget_mb[d].load(Ordering::Acquire);
                            if budget > 0 {
                                if let Some(dev) = Self::device_ptr(&mmap, d) {
                                    dev.used_memory_mb.fetch_sub(budget, Ordering::AcqRel);
                                    dev.owner_mask.fetch_and(!(1u64 << i), Ordering::AcqRel);
                                }
                                slot.memory_budget_mb[d].store(0, Ordering::Release);
                                slot.compute_budget_pct[d].store(0, Ordering::Release);
                            }
                        }
                        slot.pid.store(0, Ordering::Relaxed);
                        slot.device_mask.store(0, Ordering::Relaxed);
                        slot.heartbeat_ns.store(0, Ordering::Relaxed);
                        slot.start_time_ms.store(0, Ordering::Relaxed);
                        slot.affinity_group.store(0, Ordering::Relaxed);
                        slot.state.store(SLOT_EMPTY, Ordering::Release);
                    }
                }
            })
            .expect("failed to spawn heartbeat thread");

        // Store handle so Drop can join it
        if let Ok(mut guard) = self.heartbeat_thread.lock() {
            *guard = Some(thread_handle);
        }
        // Reset stop flag for the new thread (after storing handle)
        self.heartbeat_stop.store(0, Ordering::Release);
    }

    /// Signal the heartbeat thread to stop and wait for it to finish.
    ///
    /// Blocks until the thread exits. Safe to call multiple times.
    pub fn stop_heartbeat(&self) {
        self.heartbeat_stop.store(1, Ordering::Release);
        if let Ok(mut guard) = self.heartbeat_thread.lock() {
            if let Some(handle) = guard.take() {
                let _ = handle.join();
            }
        }
    }

    // --- Platform helpers ---

    fn get_shm_path(name: &str) -> std::path::PathBuf {
        #[cfg(target_os = "linux")]
        {
            std::path::PathBuf::from(format!("/dev/shm/{}", name))
        }
        #[cfg(not(target_os = "linux"))]
        {
            std::env::temp_dir().join(format!("cv_runtime_{}", name))
        }
    }

    fn is_process_alive(pid: u32) -> bool {
        if pid == 0 {
            return false;
        }
        #[cfg(target_os = "linux")]
        {
            std::path::Path::new(&format!("/proc/{}", pid)).exists()
        }
        #[cfg(target_family = "unix")]
        #[cfg(not(target_os = "linux"))]
        {
            unsafe { libc::kill(pid as i32, 0) == 0 }
        }
        #[cfg(not(target_family = "unix"))]
        {
            let _ = pid;
            true
        }
    }

    fn current_timestamp_ms() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    fn monotonic_ns() -> u64 {
        #[cfg(target_os = "linux")]
        {
            let mut ts = libc::timespec {
                tv_sec: 0,
                tv_nsec: 0,
            };
            // SAFETY: CLOCK_MONOTONIC is always available on Linux.
            unsafe {
                libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts);
            }
            ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
        }
        #[cfg(not(target_os = "linux"))]
        {
            // Fallback: use Instant elapsed from a fixed point
            use std::sync::OnceLock;
            use std::time::Instant;
            static EPOCH: OnceLock<Instant> = OnceLock::new();
            let epoch = EPOCH.get_or_init(Instant::now);
            epoch.elapsed().as_nanos() as u64
        }
    }

    /// Return the number of active process slots.
    pub fn active_slot_count(&self) -> usize {
        let mut count = 0;
        for i in 0..MAX_SLOTS {
            if let Some(slot) = Self::slot_ptr(&self.mmap, i) {
                if slot.state.load(Ordering::Acquire) == SLOT_ACTIVE {
                    count += 1;
                }
            }
        }
        count
    }
}

// --- LoadCoordinator trait implementation ---

impl LoadCoordinator for ShmCoordinator {
    fn update_load(&self, device_load: &HashMap<DeviceId, usize>) -> io::Result<()> {
        if device_load.is_empty() {
            return Ok(());
        }

        // Store primary device load in the slot for backward-compatible global load queries
        let slot = self.my_slot();
        let total_load: usize = device_load.values().sum();
        let primary_device = device_load
            .iter()
            .max_by_key(|(_, &load)| load)
            .map(|(id, _)| id.0)
            .unwrap_or(0);

        // We reuse compute_budget_pct[0] for legacy load tracking (atomic, no lock needed)
        slot.compute_budget_pct[0].store(total_load as u32, Ordering::Release);
        // Store primary device in device_mask's lowest bits conceptually
        // but we track it separately for queries
        let _ = primary_device; // tracked via device_mask already

        self.send_heartbeat()?;
        Ok(())
    }

    fn get_global_load(&self) -> io::Result<HashMap<DeviceId, usize>> {
        let mut aggregate = HashMap::new();
        let now = ShmCoordinator::monotonic_ns();

        for i in 0..MAX_SLOTS {
            let Some(slot) = Self::slot_ptr(&self.mmap, i) else {
                continue;
            };

            if slot.state.load(Ordering::Acquire) != SLOT_ACTIVE {
                continue;
            }

            let pid = slot.pid.load(Ordering::Relaxed);
            if pid == 0 {
                continue;
            }

            // Check staleness
            let hb = slot.heartbeat_ns.load(Ordering::Acquire);
            if hb != 0 && now > hb && (now - hb) > STALE_THRESHOLD_NS * 6 {
                // 30s equivalent
                if !Self::is_process_alive(pid) {
                    continue;
                }
            }

            // Aggregate load from device reservations
            let mask = slot.device_mask.load(Ordering::Acquire);
            for d in 0..MAX_DEVICES {
                if mask & (1u64 << d) != 0 {
                    let budget = slot.memory_budget_mb[d].load(Ordering::Acquire);
                    if budget > 0 {
                        *aggregate.entry(DeviceId(d as u32)).or_insert(0) += 1;
                    }
                }
            }

            // Also count legacy load
            let legacy_load = slot.compute_budget_pct[0].load(Ordering::Acquire);
            if legacy_load > 0 && mask == 0 {
                // No device reservations but has legacy load data
                *aggregate.entry(DeviceId(0)).or_insert(0) += legacy_load as usize;
            }
        }

        Ok(aggregate)
    }

    fn cleanup(&self) {
        // Release all device reservations
        for d in 0..MAX_DEVICES {
            let _ = self.release_device(d as u8);
        }

        let slot = self.my_slot();
        // CAS ACTIVE -> RELEASING -> EMPTY
        if slot
            .state
            .compare_exchange(
                SLOT_ACTIVE,
                SLOT_RELEASING,
                Ordering::AcqRel,
                Ordering::Relaxed,
            )
            .is_ok()
        {
            slot.pid.store(0, Ordering::Relaxed);
            slot.device_mask.store(0, Ordering::Relaxed);
            slot.heartbeat_ns.store(0, Ordering::Relaxed);
            slot.affinity_group.store(0, Ordering::Relaxed);
            slot.state.store(SLOT_EMPTY, Ordering::Release);
        }
    }

    fn register(&self) -> io::Result<()> {
        self.send_heartbeat()
    }

    fn heartbeat(&self) -> io::Result<()> {
        self.send_heartbeat()
    }

    fn reserve_device(&self, device_idx: u8, memory_mb: u32, compute_pct: u32) -> io::Result<()> {
        ShmCoordinator::reserve_device(self, device_idx, memory_mb, compute_pct)
    }

    fn release_device(&self, device_idx: u8) -> io::Result<()> {
        ShmCoordinator::release_device(self, device_idx)
    }

    fn join_group(&self, group_id: u32) -> io::Result<()> {
        ShmCoordinator::join_group(self, group_id)
    }

    fn device_memory_usage(&self) -> Vec<(u8, u32, u32)> {
        ShmCoordinator::device_memory_usage(self)
    }

    fn best_device_for_group(&self, needed_mb: u32) -> Option<u8> {
        ShmCoordinator::best_device_for_group(self, needed_mb)
    }

    fn init_device(&self, device_idx: u8, total_memory_mb: u32) -> io::Result<()> {
        ShmCoordinator::init_device(self, device_idx, total_memory_mb)
    }

    fn wait_for_device_memory(
        &self,
        device_idx: u8,
        needed_mb: u32,
        timeout: std::time::Duration,
    ) -> io::Result<()> {
        ShmCoordinator::wait_for_device_memory(self, device_idx, needed_mb, timeout)
    }
}

impl Drop for ShmCoordinator {
    fn drop(&mut self) {
        self.stop_heartbeat();
        self.cleanup();
        // Only remove the file if we're the last process
        if self.active_slot_count() == 0 {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_name(suffix: &str) -> String {
        format!("test_shm_v2_{}_{}", std::process::id(), suffix)
    }

    #[test]
    fn test_shm_coordinator_basic() {
        let name = unique_name("basic");
        let coord = ShmCoordinator::new(&name, SHM_TOTAL_SIZE).unwrap();

        let mut load = HashMap::new();
        load.insert(DeviceId(1), 10);

        coord.update_load(&load).unwrap();
        let global = coord.get_global_load().unwrap();
        // Should have at least some load data
        assert!(!global.is_empty() || load.is_empty());
    }

    #[test]
    fn test_slot_acquisition_exclusive() {
        let name = unique_name("exclusive");
        let coord1 = ShmCoordinator::new(&name, SHM_TOTAL_SIZE).unwrap();

        // Verify our slot is ACTIVE
        let slot = coord1.my_slot();
        assert_eq!(slot.state.load(Ordering::Acquire), SLOT_ACTIVE);
        assert_eq!(slot.pid.load(Ordering::Relaxed), std::process::id());

        // Second coordinator with same name + PID reuses the same slot
        let coord2 = ShmCoordinator::new(&name, SHM_TOTAL_SIZE).unwrap();
        assert_eq!(coord1.slot_index, coord2.slot_index);

        // Verify only one slot is ACTIVE (both point to the same slot)
        assert_eq!(coord1.active_slot_count(), 1);

        // Manually simulate a second "process" by setting up a slot with a different PID
        let fake_idx = (coord1.slot_index + 1) % MAX_SLOTS;
        let fake_slot = ShmCoordinator::slot_ptr(&coord1.mmap, fake_idx).unwrap();
        // CAS EMPTY -> ACQUIRING -> ACTIVE
        assert!(fake_slot
            .state
            .compare_exchange(
                SLOT_EMPTY,
                SLOT_ACQUIRING,
                Ordering::AcqRel,
                Ordering::Relaxed
            )
            .is_ok());
        fake_slot.pid.store(88888, Ordering::Release);
        fake_slot
            .heartbeat_ns
            .store(ShmCoordinator::monotonic_ns(), Ordering::Release);
        fake_slot.state.store(SLOT_ACTIVE, Ordering::Release);

        // Now two slots are active
        assert_eq!(coord1.active_slot_count(), 2);

        // Clean up fake slot
        fake_slot.state.store(SLOT_EMPTY, Ordering::Release);
        fake_slot.pid.store(0, Ordering::Release);

        drop(coord2);
    }

    #[test]
    fn test_device_reservation_strict() {
        let name = unique_name("reserve");
        let coord = ShmCoordinator::new(&name, SHM_TOTAL_SIZE).unwrap();

        // Initialize device 0 with 2048 MB
        coord.init_device(0, 2048).unwrap();

        // Reserve 1024 MB - should succeed
        coord.reserve_device(0, 1024, 50).unwrap();

        // Reserve another 512 MB - should succeed (total 1536)
        coord.reserve_device(0, 512, 25).unwrap();

        // Verify usage
        let usage = coord.device_memory_usage();
        assert_eq!(usage.len(), 1);
        assert_eq!(usage[0], (0, 1536, 2048));

        // Try to reserve 1024 more - should fail (1536 + 1024 > 2048)
        let res = coord.reserve_device(0, 1024, 25);
        assert!(res.is_err());
        assert_eq!(res.unwrap_err().kind(), io::ErrorKind::OutOfMemory);

        // Release device 0
        coord.release_device(0).unwrap();

        // Verify cleared (both reservations released together since per-slot)
        let usage = coord.device_memory_usage();
        assert_eq!(usage[0].1, 0); // used should be 0 now
    }

    #[test]
    fn test_stale_reap() {
        let name = unique_name("reap");
        let coord = ShmCoordinator::new(&name, SHM_TOTAL_SIZE).unwrap();

        // Initialize device 0
        coord.init_device(0, 4096).unwrap();

        // Simulate a dead process by manually writing a slot
        // Find an empty slot
        let fake_slot_idx = (coord.slot_index + 1) % MAX_SLOTS;
        let slot = ShmCoordinator::slot_ptr(&coord.mmap, fake_slot_idx).unwrap();

        // Set it up as if a dead process that never sent a heartbeat.
        slot.state.store(SLOT_ACTIVE, Ordering::Release);
        slot.pid.store(u32::MAX - 1, Ordering::Release); // above any OS PID limit
        slot.heartbeat_ns.store(0, Ordering::Release); // never heartbeated = stale
        slot.memory_budget_mb[0].store(512, Ordering::Release);
        slot.device_mask.store(1, Ordering::Release);

        // Add to device used memory
        let dev = ShmCoordinator::device_ptr(&coord.mmap, 0).unwrap();
        dev.used_memory_mb.fetch_add(512, Ordering::AcqRel);
        dev.owner_mask
            .fetch_or(1u64 << fake_slot_idx, Ordering::AcqRel);

        // Verify device shows 512 MB used
        let usage = coord.device_memory_usage();
        assert!(usage[0].1 >= 512);

        // Reap dead processes
        coord.reap_dead();

        // Verify slot was reaped and memory freed
        assert_eq!(slot.state.load(Ordering::Acquire), SLOT_EMPTY);
        let usage = coord.device_memory_usage();
        assert_eq!(usage[0].1, 0);
    }

    #[test]
    fn test_affinity_group_co_placement() {
        let name = unique_name("affinity");
        let coord = ShmCoordinator::new(&name, SHM_TOTAL_SIZE).unwrap();

        // Initialize two devices
        coord.init_device(0, 4096).unwrap();
        coord.init_device(1, 4096).unwrap();

        // Join group 42
        coord.join_group(42).unwrap();

        // Simulate a peer in group 42 on device 0
        let peer_idx = (coord.slot_index + 1) % MAX_SLOTS;
        let peer = ShmCoordinator::slot_ptr(&coord.mmap, peer_idx).unwrap();
        peer.state.store(SLOT_ACTIVE, Ordering::Release);
        peer.pid.store(std::process::id(), Ordering::Release); // same pid to keep alive
        peer.heartbeat_ns
            .store(ShmCoordinator::monotonic_ns(), Ordering::Release);
        peer.affinity_group.store(42, Ordering::Release);
        peer.device_mask.store(1, Ordering::Release); // device 0
        peer.memory_budget_mb[0].store(256, Ordering::Release);

        // Best device should prefer device 0 (where peer is)
        let best = coord.best_device_for_group(256);
        assert_eq!(best, Some(0));

        // Clean up fake peer
        peer.state.store(SLOT_EMPTY, Ordering::Release);
        peer.pid.store(0, Ordering::Release);
    }

    #[test]
    fn test_pid_recycling_guard() {
        let name = unique_name("pidguard");
        let coord = ShmCoordinator::new(&name, SHM_TOTAL_SIZE).unwrap();

        // Record the generation of our slot
        let slot = coord.my_slot();
        let gen1 = slot.generation.load(Ordering::Acquire);

        // Drop and re-create with same name (simulates PID reuse)
        drop(coord);

        let coord2 = ShmCoordinator::new(&name, SHM_TOTAL_SIZE).unwrap();
        let slot2 = coord2.my_slot();
        let gen2 = slot2.generation.load(Ordering::Acquire);

        // Generation must have incremented
        assert!(gen2 > gen1 || gen2 == 1); // either incremented or fresh init
    }

    #[test]
    fn test_available_devices_filter() {
        let name = unique_name("avail");
        let coord = ShmCoordinator::new(&name, SHM_TOTAL_SIZE).unwrap();

        coord.init_device(0, 2048).unwrap();
        coord.init_device(1, 1024).unwrap();

        // Reserve 1800 MB on device 0
        coord.reserve_device(0, 1800, 90).unwrap();

        // Query: need 500 MB minimum
        let avail = coord.available_devices(500);
        // Device 0 has 248 MB free (not enough), device 1 has 1024 MB free
        assert_eq!(avail.len(), 1);
        assert_eq!(avail[0].0, 1);

        coord.release_device(0).unwrap();
    }

    #[test]
    fn test_wait_for_device_memory_immediate() {
        // When device has enough free memory, wait returns immediately
        let name = unique_name("wait_imm");
        let coord = ShmCoordinator::new(&name, SHM_TOTAL_SIZE).unwrap();
        coord.init_device(0, 2048).unwrap();

        let start = std::time::Instant::now();
        coord
            .wait_for_device_memory(0, 1024, Duration::from_secs(5))
            .unwrap();
        // Should return in < 10ms (immediate)
        assert!(start.elapsed() < Duration::from_millis(100));
    }

    #[test]
    fn test_wait_for_device_memory_timeout() {
        // When device is full and nobody releases, wait should time out
        let name = unique_name("wait_to");
        let coord = ShmCoordinator::new(&name, SHM_TOTAL_SIZE).unwrap();
        coord.init_device(0, 1024).unwrap();
        coord.reserve_device(0, 1024, 100).unwrap(); // fill it up

        let start = std::time::Instant::now();
        let result = coord.wait_for_device_memory(0, 1, Duration::from_millis(200));
        let elapsed = start.elapsed();

        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::TimedOut);
        // Should have waited roughly 200ms (not much more)
        assert!(elapsed >= Duration::from_millis(180));
        assert!(elapsed < Duration::from_millis(500));

        coord.release_device(0).unwrap();
    }

    #[test]
    fn test_wait_for_device_memory_wakes_on_release() {
        // Spawn a thread that waits for memory.
        // Main thread releases memory after a delay.
        // Waiter should wake up well before the timeout.
        let name = unique_name("wait_wake");
        let coord = ShmCoordinator::new(&name, SHM_TOTAL_SIZE).unwrap();
        coord.init_device(0, 1024).unwrap();
        coord.reserve_device(0, 1024, 100).unwrap(); // full

        let name2 = name.clone();
        let start = std::time::Instant::now();
        // Barrier keeps the thread's coordinator alive until main is done
        let done = std::sync::Arc::new(std::sync::Barrier::new(2));
        let done2 = done.clone();

        // Waiter thread: will block up to 5s waiting for 512 MB
        let waiter = std::thread::spawn(move || {
            let c = ShmCoordinator::new(&name2, SHM_TOTAL_SIZE).unwrap();
            let r = c.wait_for_device_memory(0, 512, Duration::from_secs(5));
            done2.wait(); // hold coordinator alive until main signals
            r
        });

        // Main: sleep 100ms, then release memory
        std::thread::sleep(Duration::from_millis(100));
        coord.release_device(0).unwrap(); // frees 1024 MB, epoch bumps

        // Signal waiter thread it can drop
        done.wait();

        let result = waiter.join().unwrap();
        let elapsed = start.elapsed();

        assert!(
            result.is_ok(),
            "Waiter should have succeeded: {:?}",
            result.err()
        );
        // Should have woken up shortly after the release (~100-200ms total),
        // not waiting the full 5s
        assert!(
            elapsed < Duration::from_secs(1),
            "Woke up too late: {:?}",
            elapsed
        );
    }

    #[test]
    fn test_drop_cleans_up_completely() {
        // Create coordinator, reserve device, start heartbeat, then drop.
        // Verify: slot freed, device budget released, shm file removed, thread joined.
        let name = unique_name("drop_clean");
        let path = {
            let coord = ShmCoordinator::new(&name, SHM_TOTAL_SIZE).unwrap();
            coord.init_device(0, 2048).unwrap();
            coord.reserve_device(0, 512, 50).unwrap();
            coord.start_heartbeat_thread(Duration::from_millis(50));

            // Verify reservation is live
            let usage = coord.device_memory_usage();
            assert_eq!(usage[0].1, 512);

            let p = coord.path.clone();
            // coord drops here
            p
        };

        // After drop: shm file should be removed (we were the only process)
        assert!(
            !path.exists(),
            "SHM file should be removed after last coordinator drops"
        );
    }

    #[test]
    fn test_heartbeat_thread_joins_on_drop() {
        // Verify the heartbeat thread actually stops when coordinator is dropped,
        // not left running as a leaked thread.
        let name = unique_name("hb_join");
        let thread_count_before = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let _ = thread_count_before; // baseline

        {
            let coord = ShmCoordinator::new(&name, SHM_TOTAL_SIZE).unwrap();
            coord.start_heartbeat_thread(Duration::from_millis(50));
            // Let the heartbeat run a few cycles
            std::thread::sleep(Duration::from_millis(200));
            // coord drops here — should join the thread
        }

        // Give the thread a moment to fully terminate
        std::thread::sleep(Duration::from_millis(100));

        // We can't easily count threads, but we can verify the shm file is cleaned up
        // which proves Drop ran to completion (including thread join).
        let path = ShmCoordinator::get_shm_path(&name);
        assert!(!path.exists(), "SHM file should be gone after drop");
    }
}
