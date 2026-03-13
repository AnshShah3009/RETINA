use super::LoadCoordinator;
use cv_hal::DeviceId;
use std::collections::HashMap;
use std::io;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

const MAX_PROCESSES: usize = 64;

#[repr(C)]
struct LoadSlot {
    lock: AtomicU32,
    device_id: AtomicU32,
    load: AtomicU32,
    timestamp: AtomicU64,
    pid: AtomicU32,
}

impl Default for LoadSlot {
    fn default() -> Self {
        Self {
            lock: AtomicU32::new(0),
            device_id: AtomicU32::new(0),
            load: AtomicU32::new(0),
            timestamp: AtomicU64::new(0),
            pid: AtomicU32::new(0),
        }
    }
}

impl LoadSlot {
    fn try_lock(&self) -> bool {
        self.lock
            .compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }

    fn unlock(&self) {
        self.lock.store(0, Ordering::Release);
    }

    fn with_lock<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&Self) -> R,
    {
        let mut spins = 0;
        while !self.try_lock() {
            spins += 1;
            if spins > 1000 {
                std::hint::spin_loop();
            }
            std::thread::yield_now();
        }
        let result = f(self);
        self.unlock();
        result
    }
}

#[repr(C)]
struct ShmHeader {
    magic: AtomicU32,
    version: AtomicU32,
    num_slots: AtomicU32,
    slot_size: AtomicU32,
}

pub struct ShmCoordinator {
    mmap: memmap2::MmapMut,
    slot_index: usize,
    #[allow(dead_code)]
    pid: u32,
    path: std::path::PathBuf,
}

impl ShmCoordinator {
    pub fn new(name: &str, size: usize) -> io::Result<Self> {
        let path = Self::get_shm_path(name);

        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;

        file.set_len(size as u64)?;

        let mut mmap = unsafe { memmap2::MmapOptions::new().map_mut(&file)? };

        for byte in mmap.iter_mut() {
            *byte = 0;
        }

        let header = Self::header_mut(&mut mmap).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "Mmap too small for shared memory header",
            )
        })?;
        let magic = 0x43565254;

        if header.magic.load(Ordering::Relaxed) != magic {
            header.magic.store(magic, Ordering::Relaxed);
            header.version.store(1, Ordering::Relaxed);
            header
                .num_slots
                .store(MAX_PROCESSES as u32, Ordering::Relaxed);
            header
                .slot_size
                .store(std::mem::size_of::<LoadSlot>() as u32, Ordering::Relaxed);
        }

        let pid = std::process::id();
        let slot_index = Self::find_or_allocate_slot(&mut mmap, pid)?;

        Ok(Self {
            mmap,
            slot_index,
            pid,
            path,
        })
    }

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

    /// Get the header from the mmap with bounds validation.
    ///
    /// Returns None if the mmap is too small for the header.
    fn header_mut(mmap: &mut memmap2::MmapMut) -> Option<&mut ShmHeader> {
        let header_size = std::mem::size_of::<ShmHeader>();
        if mmap.len() < header_size {
            return None;
        }
        Some(unsafe { &mut *(mmap.as_mut_ptr() as *mut ShmHeader) })
    }

    /// Get a slot at the given index with bounds validation (mutable mmap).
    ///
    /// # Safety
    /// The mmap must be large enough to contain the slot at the given index.
    fn get_slot_mut_at(mmap: &mut memmap2::MmapMut, index: usize) -> Option<&mut LoadSlot> {
        let header_size = std::mem::size_of::<ShmHeader>();
        let slot_size = std::mem::size_of::<LoadSlot>();
        let offset = header_size.checked_add(index.checked_mul(slot_size)?)?;

        if offset + slot_size > mmap.len() {
            return None;
        }

        Some(unsafe { &mut *(mmap.as_mut_ptr().add(offset) as *mut LoadSlot) })
    }

    fn find_or_allocate_slot(mmap: &mut memmap2::MmapMut, pid: u32) -> io::Result<usize> {
        let header = Self::header_mut(mmap).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "Mmap too small for shared memory header",
            )
        })?;
        let num_slots = header.num_slots.load(Ordering::Relaxed) as usize;

        for i in 0..num_slots.min(MAX_PROCESSES) {
            let slot = Self::get_slot_mut_at(mmap, i)
                .ok_or_else(|| io::Error::other("Invalid slot offset in shared memory"))?;

            let slot_pid = slot.pid.load(Ordering::Relaxed);

            if slot_pid == pid {
                return Ok(i);
            }

            if (slot_pid == 0 || !Self::is_process_alive(slot_pid)) && slot.try_lock() {
                let current_pid = slot.pid.load(Ordering::Relaxed);
                if current_pid == 0 || !Self::is_process_alive(current_pid) {
                    slot.pid.store(pid, Ordering::Relaxed);
                    slot.unlock();
                    return Ok(i);
                }
                slot.unlock();
            }
        }

        Err(io::Error::other(
            "No available slots in shared memory coordinator",
        ))
    }

    /// Get the slot at the stored index with bounds validation.
    fn get_slot(&self) -> &LoadSlot {
        let header_size = std::mem::size_of::<ShmHeader>();
        let slot_size = std::mem::size_of::<LoadSlot>();
        let offset = header_size + self.slot_index * slot_size;

        debug_assert!(
            offset + slot_size <= self.mmap.len(),
            "Slot offset out of bounds: {} + {} > {}",
            offset,
            slot_size,
            self.mmap.len()
        );

        unsafe { &*(self.mmap.as_ptr().add(offset) as *const LoadSlot) }
    }

    fn is_process_alive(pid: u32) -> bool {
        #[cfg(target_os = "linux")]
        {
            std::path::Path::new(&format!("/proc/{}", pid)).exists()
        }
        #[cfg(target_family = "unix")]
        #[cfg(not(target_os = "linux"))]
        {
            // Use libc::kill with signal 0 to check if process exists on macOS/BSD.
            // Returns true if process is alive, false if it doesn't exist.
            unsafe { libc::kill(pid as i32, 0) == 0 }
        }
        #[cfg(not(target_family = "unix"))]
        {
            // Windows: conservatively assume process is alive to avoid premature slot reuse.
            // A stale timeout will eventually clean up if the process is gone.
            let _ = pid;
            true
        }
    }

    fn current_timestamp() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Get the header from the mmap (read-only) with bounds validation.
    fn header(&self) -> io::Result<&ShmHeader> {
        let header_size = std::mem::size_of::<ShmHeader>();
        if self.mmap.len() < header_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Mmap too small for shared memory header",
            ));
        }
        Ok(unsafe { &*(self.mmap.as_ptr() as *const ShmHeader) })
    }

    /// Get a slot at the given index (read-only) with bounds validation.
    fn get_slot_at(&self, index: usize) -> Option<&LoadSlot> {
        let header_size = std::mem::size_of::<ShmHeader>();
        let slot_size = std::mem::size_of::<LoadSlot>();
        let offset = header_size.checked_add(index.checked_mul(slot_size)?)?;

        if offset + slot_size > self.mmap.len() {
            return None;
        }

        Some(unsafe { &*(self.mmap.as_ptr().add(offset) as *const LoadSlot) })
    }
}

impl LoadCoordinator for ShmCoordinator {
    fn update_load(&self, device_load: &HashMap<DeviceId, usize>) -> std::io::Result<()> {
        if device_load.is_empty() {
            return Ok(());
        }

        let timestamp = Self::current_timestamp();
        let total_load: usize = device_load.values().sum();
        let primary_device = device_load
            .iter()
            .max_by_key(|(_, &load)| load)
            .map(|(id, _)| id.0)
            .unwrap_or(0);

        let slot = self.get_slot();
        slot.with_lock(|s| {
            s.device_id.store(primary_device, Ordering::Relaxed);
            s.load.store(total_load as u32, Ordering::Relaxed);
            s.timestamp.store(timestamp, Ordering::Relaxed);
        });

        Ok(())
    }

    fn get_global_load(&self) -> std::io::Result<HashMap<DeviceId, usize>> {
        let mut aggregate = HashMap::new();
        let now = Self::current_timestamp();
        let stale_timeout = 30_000u64;

        let header = self.header()?;
        let num_slots = header.num_slots.load(Ordering::Relaxed) as usize;

        for i in 0..num_slots {
            let Some(slot) = self.get_slot_at(i) else {
                continue;
            };

            let slot_pid = slot.pid.load(Ordering::Relaxed);
            if slot_pid == 0 {
                continue;
            }

            let timestamp = slot.timestamp.load(Ordering::Relaxed);
            // Skip if timestamp is in the future (clock skew).
            // Skip if timestamp is stale and process is dead.
            // Keep if data is recent or process is still alive.
            let is_stale = timestamp <= now && (now - timestamp) > stale_timeout;
            if is_stale && !Self::is_process_alive(slot_pid) {
                continue;
            }

            let device_id = slot.device_id.load(Ordering::Relaxed);
            let load = slot.load.load(Ordering::Relaxed);

            if device_id != 0 {
                *aggregate.entry(DeviceId(device_id)).or_insert(0) += load as usize;
            }
        }

        Ok(aggregate)
    }

    fn cleanup(&self) {
        let slot = self.get_slot();
        slot.with_lock(|s| {
            s.pid.store(0, Ordering::Relaxed);
            s.device_id.store(0, Ordering::Relaxed);
            s.load.store(0, Ordering::Relaxed);
            s.timestamp.store(0, Ordering::Relaxed);
        });
    }
}

impl Drop for ShmCoordinator {
    fn drop(&mut self) {
        self.cleanup();
        let _ = std::fs::remove_file(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shm_coordinator() {
        let name = format!("test_shm_{}", std::process::id());
        let coord = ShmCoordinator::new(&name, 4096).unwrap();

        let mut load = HashMap::new();
        load.insert(DeviceId(1), 10);

        coord.update_load(&load).unwrap();

        let global = coord.get_global_load().unwrap();
        assert!(global.contains_key(&DeviceId(1)));
    }
}
