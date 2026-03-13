use crate::device_registry::{registry, DeviceRuntime};
use crate::distributed::{FileCoordinator, LoadCoordinator};
use crate::executor::{Executor, ExecutorConfig};
use crate::Result;
use cv_hal::{BackendType, DeviceId};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

/// Timeout for device recovery after failure. If a device has been marked as failed
/// for longer than this period, it will be re-tried.
const DEVICE_RECOVERY_TIMEOUT: Duration = Duration::from_secs(60);

/// Priority level for a resource group
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Background = 0,
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
}

/// Hints for the scheduler to select the most appropriate resource group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadHint {
    /// Latency-sensitive (e.g., UI, real-time tracking). Prefers faster start times.
    Latency,
    /// Throughput-oriented (e.g., batch processing, dense mapping). Prefers throughput.
    Throughput,
    /// Power-saving (e.g., background tasks).
    PowerSave,
    /// Default behavior.
    Default,
}

/// Statistics for a workload across all coordinated processes.
#[derive(Debug, Clone, Default)]
pub struct WorkloadStats {
    pub active_tasks: usize,
    pub total_groups: usize,
}

/// The mode of the orchestrator: either standalone or coordinated via a daemon.
#[derive(Debug, Clone)]
pub enum OrchestratorMode {
    /// Local-only orchestration.
    Local,
    /// Distributed orchestration via a central coordinator (Hybrid mode).
    /// Uses a shared filesystem entry for basic lock-based coordination.
    Distributed {
        coordinator_path: std::path::PathBuf,
    },
}

/// Policy for a resource group
#[derive(Debug, Clone, Copy)]
pub struct GroupPolicy {
    /// If true, this group uses the global thread pool (work stealing enabled)
    pub allow_work_stealing: bool,
    /// If true, the pool can be resized at runtime
    pub allow_dynamic_scaling: bool,
    /// Priority level for tasks in this group
    pub priority: TaskPriority,
}

impl Default for GroupPolicy {
    fn default() -> Self {
        Self {
            allow_work_stealing: true,
            allow_dynamic_scaling: true,
            priority: TaskPriority::Normal,
        }
    }
}

#[derive(Debug)]
pub struct ResourceGroup {
    pub name: String,
    pub policy: GroupPolicy,
    device_id: DeviceId,
    pub(crate) executor: Arc<Executor>,
    core_ids: Option<Vec<usize>>,
}

impl ResourceGroup {
    pub fn new(
        name: &str,
        device_id: DeviceId,
        num_threads: usize,
        core_ids: Option<Vec<usize>>,
        policy: GroupPolicy,
    ) -> Result<Self> {
        let config = ExecutorConfig {
            num_threads,
            name: name.to_string(),
            work_stealing: policy.allow_work_stealing,
            core_affinity: core_ids.clone(),
        };

        let executor = Arc::new(Executor::with_config(device_id, config)?);

        Ok(Self {
            name: name.to_string(),
            policy,
            device_id,
            executor,
            core_ids,
        })
    }

    pub fn spawn<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.executor.spawn(f);
    }

    pub fn run<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        self.executor.install(f)
    }

    pub fn device_id(&self) -> DeviceId {
        self.device_id
    }

    pub fn load(&self) -> usize {
        self.executor.load()
    }

    pub fn num_threads(&self) -> usize {
        self.executor.num_threads()
    }

    pub fn core_ids(&self) -> Option<&[usize]> {
        self.core_ids.as_deref()
    }

    pub fn device_runtime(&self) -> Result<Arc<DeviceRuntime>> {
        registry()?.get_device(self.device_id).ok_or_else(|| {
            crate::Error::RuntimeError(format!("Device {:?} not found", self.device_id))
        })
    }

    /// Get the compute device for this resource group
    /// Returns a Result instead of panicking on error
    pub fn device(&self) -> Result<cv_hal::compute::ComputeDevice<'static>> {
        self.try_device()
    }

    pub fn try_device(&self) -> Result<cv_hal::compute::ComputeDevice<'static>> {
        cv_hal::compute::get_device_by_id(self.device_id).map_err(|e| {
            crate::Error::RuntimeError(format!(
                "Could not find compute device {:?}: {}",
                self.device_id, e
            ))
        })
    }

    pub fn resize(&self, new_num_threads: usize) -> Result<()> {
        if !self.policy.allow_dynamic_scaling {
            return Err(crate::Error::RuntimeError(format!(
                "Resource group '{}' does not allow dynamic scaling",
                self.name
            )));
        }
        self.executor.resize(new_num_threads)
    }

    pub fn set_core_affinity(&self, cores: Vec<usize>) -> Result<()> {
        self.executor.set_core_affinity(cores)
    }
}

pub struct TaskScheduler {
    groups: Mutex<HashMap<String, Arc<ResourceGroup>>>,
    mode: OrchestratorMode,
    coordinator: Option<Box<dyn LoadCoordinator>>,
    global_load_cache: Mutex<(HashMap<DeviceId, usize>, Instant)>,
    /// Maps failed devices to the time of failure. Devices are considered recovered
    /// after DEVICE_RECOVERY_TIMEOUT has elapsed.
    failed_devices: Mutex<HashMap<DeviceId, Instant>>,
}

pub enum RuntimeRunner {
    Group(Arc<ResourceGroup>),
    Sync(DeviceId),
}

impl RuntimeRunner {
    pub fn run<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        match self {
            RuntimeRunner::Group(g) => g.run(f),
            RuntimeRunner::Sync(_) => f(),
        }
    }

    /// Run with automatic fallback to another device on error
    pub fn run_safe<F, R, E>(&self, f: F) -> std::result::Result<R, E>
    where
        F: Fn() -> std::result::Result<R, E> + Send + Clone,
        R: Send,
        E: From<crate::Error> + Send,
    {
        let f_cloned = f.clone();
        let res = self.run(f_cloned);
        if res.is_err() {
            // Signal failure to scheduler for future calls
            if let Ok(s) = scheduler() {
                s.report_failure(self.device_id());
            }
            // Fallback to CPU immediately
            let reg = registry().map_err(E::from)?;
            let cpu_id = reg.default_cpu().id();
            if cpu_id != self.device_id() {
                return f();
            }
        }
        res
    }

    pub fn device_id(&self) -> DeviceId {
        match self {
            RuntimeRunner::Group(g) => g.device_id(),
            RuntimeRunner::Sync(id) => *id,
        }
    }

    pub fn device_runtime(&self) -> Result<Arc<DeviceRuntime>> {
        registry()?.get_device(self.device_id()).ok_or_else(|| {
            crate::Error::RuntimeError(format!("Device {:?} not found", self.device_id()))
        })
    }

    /// Get the compute device for this runner
    /// Returns a Result instead of panicking on error
    pub fn device(&self) -> Result<cv_hal::compute::ComputeDevice<'static>> {
        self.try_device()
    }

    pub fn try_device(&self) -> Result<cv_hal::compute::ComputeDevice<'static>> {
        cv_hal::compute::get_device_by_id(self.device_id()).map_err(|e| {
            crate::Error::RuntimeError(format!(
                "Could not find compute device {:?}: {}",
                self.device_id(),
                e
            ))
        })
    }
}

/// Get the best available runtime runner (GPU if available, else CPU)
/// Returns a Result instead of panicking on error
pub fn best_runner() -> Result<RuntimeRunner> {
    try_best_runner()
}

pub fn try_best_runner() -> Result<RuntimeRunner> {
    if let Ok(s) = scheduler() {
        if let Ok(g) = s.best_gpu_or_cpu() {
            return Ok(RuntimeRunner::Group(g));
        }
    }

    if let Ok(reg) = registry() {
        return Ok(RuntimeRunner::Sync(reg.default_cpu().id()));
    }

    Err(crate::Error::RuntimeError(
        "Could not initialize even basic device registry".into(),
    ))
}

/// Get the default runtime runner (from configured default group or CPU fallback)
/// Returns a Result instead of panicking on error
pub fn default_runner() -> Result<RuntimeRunner> {
    try_default_runner()
}

pub fn try_default_runner() -> Result<RuntimeRunner> {
    if let Ok(s) = scheduler() {
        if let Ok(g) = s.get_default_group() {
            return Ok(RuntimeRunner::Group(g));
        }
    }

    if let Ok(reg) = registry() {
        return Ok(RuntimeRunner::Sync(reg.default_cpu().id()));
    }

    Err(crate::Error::RuntimeError(
        "Could not initialize even basic device registry".into(),
    ))
}

impl Default for TaskScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskScheduler {
    pub fn new() -> Self {
        let (mode, coordinator): (OrchestratorMode, Option<Box<dyn LoadCoordinator>>) =
            if let Ok(path) = std::env::var("CV_RUNTIME_COORDINATOR") {
                let path_buf = std::path::PathBuf::from(path);
                (
                    OrchestratorMode::Distributed {
                        coordinator_path: path_buf.clone(),
                    },
                    Some(Box::new(FileCoordinator::new(path_buf))),
                )
            } else {
                (OrchestratorMode::Local, None)
            };

        Self {
            groups: Mutex::new(HashMap::new()),
            mode,
            coordinator,
            global_load_cache: Mutex::new((
                HashMap::new(),
                Instant::now() - Duration::from_secs(3600),
            )),
            failed_devices: Mutex::new(HashMap::new()),
        }
    }

    pub fn mode(&self) -> &OrchestratorMode {
        &self.mode
    }

    pub fn report_failure(&self, device_id: DeviceId) {
        let mut failed = self.failed_devices.lock();
        failed.insert(device_id, Instant::now());
    }

    pub fn is_device_healthy(&self, device_id: DeviceId) -> bool {
        let mut failed = self.failed_devices.lock();

        // If device has never failed, it's healthy
        match failed.get(&device_id) {
            None => true,
            Some(&failure_time) => {
                // If device failed but has recovered (timeout elapsed), mark it as healthy
                if failure_time.elapsed() > DEVICE_RECOVERY_TIMEOUT {
                    failed.remove(&device_id);
                    true
                } else {
                    false
                }
            }
        }
    }

    pub fn create_group(
        &self,
        name: &str,
        num_threads: usize,
        cores: Option<Vec<usize>>,
        policy: GroupPolicy,
    ) -> Result<Arc<ResourceGroup>> {
        let cpu_id = registry()?.default_cpu().id();
        self.create_group_with_device(name, num_threads, cores, policy, cpu_id)
    }

    pub fn create_group_with_device(
        &self,
        name: &str,
        num_threads: usize,
        cores: Option<Vec<usize>>,
        policy: GroupPolicy,
        device_id: DeviceId,
    ) -> Result<Arc<ResourceGroup>> {
        let mut groups = self.groups.lock();

        if groups.contains_key(name) {
            return Err(crate::Error::RuntimeError(format!(
                "Resource group '{}' already exists",
                name
            )));
        }

        let group = Arc::new(ResourceGroup::new(
            name,
            device_id,
            num_threads,
            cores,
            policy,
        )?);
        groups.insert(name.to_string(), group.clone());

        // Also register with the device runtime
        if let Some(runtime) = registry()?.get_device(device_id) {
            runtime
                .executors()
                .lock()
                .map_err(|_| {
                    crate::Error::ConcurrencyError(
                        "Device runtime executor pool lock poisoned".into(),
                    )
                })?
                .add_executor(group.executor.clone());
        }

        Ok(group)
    }

    pub fn remove_group(&self, name: &str) -> Result<Option<Arc<ResourceGroup>>> {
        let mut groups = self.groups.lock();
        if let Some(group) = groups.remove(name) {
            Ok(Some(group))
        } else {
            Ok(None)
        }
    }

    pub fn get_group(&self, name: &str) -> Result<Option<Arc<ResourceGroup>>> {
        let groups = self.groups.lock();
        Ok(groups.get(name).cloned())
    }

    fn get_global_load(&self) -> HashMap<DeviceId, usize> {
        let mut cache = self.global_load_cache.lock();
        if cache.1.elapsed() > Duration::from_millis(200) {
            if let Some(ref coord) = self.coordinator {
                // Periodically update our own load too
                let local_load = self.get_local_load();
                let _ = coord.update_load(&local_load);

                if let Ok(global) = coord.get_global_load() {
                    cache.0 = global;
                }
            }
            cache.1 = Instant::now();
        }
        cache.0.clone()
    }

    fn get_local_load(&self) -> HashMap<DeviceId, usize> {
        let groups = self.groups.lock();
        let mut load = HashMap::new();
        for group in groups.values() {
            let entry = load.entry(group.device_id()).or_insert(0);
            *entry += group.load();
        }
        load
    }

    /// Finds the best available resource group for a given device type.
    /// Prefers groups with higher priority, then those with the least active tasks.
    pub fn get_best_group(
        &self,
        backend_type: BackendType,
        hint: WorkloadHint,
    ) -> Result<Option<Arc<ResourceGroup>>> {
        let global_load = self.get_global_load();
        let groups = self.groups.lock();

        // Pre-compute local load per device to avoid O(n²) in the loop
        let local_load_by_device: HashMap<DeviceId, usize> = {
            let mut map = HashMap::new();
            for group in groups.values() {
                *map.entry(group.device_id()).or_insert(0) += group.load();
            }
            map
        };

        let mut best_group: Option<Arc<ResourceGroup>> = None;
        let mut max_priority = TaskPriority::Background;
        let mut min_load = usize::MAX;

        for group in groups.values() {
            let device_id = group.device_id();
            if !self.is_device_healthy(device_id) {
                continue;
            }

            let runtime = group.device_runtime()?;
            let matches = match (backend_type, runtime.backend()) {
                (BackendType::Cpu, BackendType::Cpu) => true,
                // Any GPU backend matches a GPU device for now
                (t, b) if t != BackendType::Cpu && b != BackendType::Cpu => true,
                _ => false,
            };

            if matches {
                let priority = group.policy.priority;

                // Adjust selection logic based on hint
                if hint == WorkloadHint::Latency && priority < TaskPriority::Normal {
                    continue; // Skip low priority for latency sensitive
                }

                // Load calculation: Local device load + remote load for this device
                let local_device_load = local_load_by_device.get(&device_id).copied().unwrap_or(0);
                let remote_load = global_load
                    .get(&device_id)
                    .copied()
                    .unwrap_or(0)
                    .saturating_sub(local_device_load);
                let total_device_load = group.load() + remote_load;

                if priority > max_priority {
                    max_priority = priority;
                    min_load = total_device_load;
                    best_group = Some(group.clone());
                } else if priority == max_priority && total_device_load < min_load {
                    min_load = total_device_load;
                    best_group = Some(group.clone());
                }
            }
        }

        Ok(best_group)
    }

    pub fn get_default_group(&self) -> Result<Arc<ResourceGroup>> {
        self.get_group("default")?.ok_or_else(|| {
            crate::Error::RuntimeError("Default resource group not found".to_string())
        })
    }

    pub fn best_gpu_or_cpu(&self) -> Result<Arc<ResourceGroup>> {
        self.best_gpu_or_cpu_for(WorkloadHint::Default)
    }

    pub fn best_gpu_or_cpu_for(&self, hint: WorkloadHint) -> Result<Arc<ResourceGroup>> {
        // Try WebGPU first (as it's our primary accelerator)
        if let Some(group) = self.get_best_group(BackendType::WebGPU, hint)? {
            return Ok(group);
        }
        if let Some(group) = self.get_best_group(BackendType::Vulkan, hint)? {
            return Ok(group);
        }
        self.get_default_group()
    }

    pub fn submit<F>(&self, group_name: &str, f: F) -> Result<()>
    where
        F: FnOnce() + Send + 'static,
    {
        if let Some(group) = self.get_group(group_name)? {
            group.spawn(f);
            Ok(())
        } else {
            Err(crate::Error::RuntimeError(format!(
                "Resource group '{}' not found",
                group_name
            )))
        }
    }
}

static GLOBAL_SCHEDULER: OnceLock<Result<TaskScheduler>> = OnceLock::new();

pub fn scheduler() -> Result<&'static TaskScheduler> {
    GLOBAL_SCHEDULER
        .get_or_init(|| {
            let s = TaskScheduler::new();
            s.create_group("default", num_cpus::get(), None, GroupPolicy::default())?;
            Ok(s)
        })
        .as_ref()
        .map_err(|e| crate::Error::RuntimeError(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_unified_concurrency() {
        let s = TaskScheduler::new();
        let policy = GroupPolicy::default();

        let g1 = s.create_group("g1", 4, None, policy).unwrap();
        let g2 = s.create_group("g2", 2, None, policy).unwrap();

        assert_eq!(g1.executor.load(), 0);
        assert_eq!(g2.executor.load(), 0);
    }

    #[test]
    fn test_duplicate_group_error() {
        let s = TaskScheduler::new();
        let policy = GroupPolicy::default();

        s.create_group("same", 2, None, policy).unwrap();
        let res = s.create_group("same", 2, None, policy);

        assert!(res.is_err());
        assert!(res.unwrap_err().to_string().contains("already exists"));
    }

    #[test]
    fn test_submit_logic() {
        let s = TaskScheduler::new();
        s.create_group("worker", 2, None, GroupPolicy::default())
            .unwrap();

        let (tx, rx) = std::sync::mpsc::channel();
        s.submit("worker", move || {
            tx.send(42).unwrap();
        })
        .unwrap();

        assert_eq!(rx.recv_timeout(Duration::from_secs(1)).unwrap(), 42);
    }

    #[test]
    fn test_load_aware_steering() {
        let s = TaskScheduler::new();
        let policy = GroupPolicy::default();

        let cpu_id = registry().unwrap().default_cpu().id();
        let g1 = s
            .create_group_with_device("g1", 2, None, policy, cpu_id)
            .unwrap();
        let g2 = s
            .create_group_with_device("g2", 2, None, policy, cpu_id)
            .unwrap();

        assert_eq!(g1.load(), 0);
        assert_eq!(g2.load(), 0);

        let (tx, rx) = std::sync::mpsc::channel();
        let g1_cloned = g1.clone();
        std::thread::spawn(move || {
            g1_cloned.run(|| {
                tx.send(()).unwrap();
                std::thread::sleep(Duration::from_millis(200));
            });
        });

        rx.recv().unwrap();
        // Wait a bit for load to update
        std::thread::sleep(Duration::from_millis(50));
        assert!(g1.load() >= 1);

        let best = s
            .get_best_group(BackendType::Cpu, WorkloadHint::Default)
            .unwrap()
            .unwrap();
        assert_eq!(best.name, "g2");
    }
}
