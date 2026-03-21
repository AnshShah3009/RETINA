use crate::device_registry::{registry, DeviceRuntime};
use crate::distributed::{FileCoordinator, LoadCoordinator, ShmCoordinator, SHM_SIZE};
use crate::executor::{Executor, ExecutorConfig};
use crate::Result;
use cv_hal::{BackendType, DeviceId};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use tracing::warn;

/// Timeout for device recovery after failure. If a device has been marked as failed
/// for longer than this period, it will be re-tried.
const DEVICE_RECOVERY_TIMEOUT: Duration = Duration::from_secs(60);

/// Default timeout for waiting on GPU VRAM (30 seconds).
/// Override system-wide with `CV_GPU_WAIT_TIMEOUT_MS`.
const DEFAULT_GPU_WAIT_TIMEOUT: Duration = Duration::from_secs(30);

/// Read the GPU wait timeout from env or return the default.
fn gpu_wait_timeout() -> Duration {
    std::env::var("CV_GPU_WAIT_TIMEOUT_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(Duration::from_millis)
        .unwrap_or(DEFAULT_GPU_WAIT_TIMEOUT)
}

/// Read the load cache interval from env or return the default (200ms).
fn load_cache_interval() -> Duration {
    std::env::var("CV_RUNTIME_LOAD_CACHE_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(Duration::from_millis)
        .unwrap_or(Duration::from_millis(200))
}

/// Priority level for scheduling tasks within resource groups.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    /// Lowest priority, suitable for non-urgent background work.
    Background = 0,
    /// Below-normal priority.
    Low = 1,
    /// Default priority for most workloads.
    Normal = 2,
    /// Elevated priority for latency-sensitive work.
    High = 3,
    /// Highest priority, reserved for must-complete operations.
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
    /// Must run on a specific backend (e.g. Vulkan, WebGPU).
    Require(BackendType),
    /// Prefers a specific backend but can fall back.
    Prefer(BackendType),
    /// Default behavior.
    Default,
}

/// Aggregate workload statistics across all coordinated processes.
#[derive(Debug, Clone, Default)]
pub struct WorkloadStats {
    /// Number of tasks currently executing or queued.
    pub active_tasks: usize,
    /// Total number of resource groups across all processes.
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

/// Scheduling and scaling policy for a [`ResourceGroup`].
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

/// A named group of threads bound to a single device, used for executing tasks.
///
/// Resource groups isolate workloads and allow per-group scheduling policies,
/// core pinning, and dynamic resizing.
#[derive(Debug)]
pub struct ResourceGroup {
    /// Human-readable name (e.g. `"default"`, `"gpu-0"`).
    pub name: String,
    /// Scheduling policy for this group.
    pub policy: GroupPolicy,
    device_id: DeviceId,
    /// Cached at creation so `get_best_group` doesn't hit the registry mutex per group.
    backend: BackendType,
    pub(crate) executor: Arc<Executor>,
    core_ids: Option<Vec<usize>>,
}

impl ResourceGroup {
    /// Create a new resource group with the given thread count and policy.
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

        // Cache backend type so get_best_group() doesn't hit the registry mutex.
        let backend = registry()
            .ok()
            .and_then(|r| r.get_device(device_id))
            .map(|d| d.backend())
            .unwrap_or(BackendType::Cpu);

        Ok(Self {
            name: name.to_string(),
            policy,
            device_id,
            backend,
            executor,
            core_ids,
        })
    }

    /// Spawn a fire-and-forget task on this group's thread pool.
    pub fn spawn<F>(&self, f: F) -> crate::Result<()>
    where
        F: FnOnce() + Send + 'static,
    {
        self.executor.spawn(f);
        Ok(())
    }

    /// Run a closure synchronously on this group's thread pool and return its result.
    pub fn run<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        self.executor.install(f)
    }

    /// Return the device this group is bound to.
    pub fn device_id(&self) -> DeviceId {
        self.device_id
    }

    /// Return the cached backend type (no registry mutex hit).
    pub fn backend(&self) -> BackendType {
        self.backend
    }

    /// Return the number of in-flight tasks on this group.
    pub fn load(&self) -> usize {
        self.executor.load()
    }

    /// Return the number of threads in this group's pool.
    pub fn num_threads(&self) -> usize {
        self.executor.num_threads()
    }

    /// Return the core IDs this group is pinned to, if any.
    pub fn core_ids(&self) -> Option<&[usize]> {
        self.core_ids.as_deref()
    }

    /// Look up this group's [`DeviceRuntime`] in the global registry.
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

    /// Try to resolve the HAL compute device for this group.
    pub fn try_device(&self) -> Result<cv_hal::compute::ComputeDevice<'static>> {
        cv_hal::compute::get_device_by_id(self.device_id).map_err(|e| {
            crate::Error::RuntimeError(format!(
                "Could not find compute device {:?}: {}",
                self.device_id, e
            ))
        })
    }

    /// Resize the thread pool. Fails if `allow_dynamic_scaling` is false.
    pub fn resize(&self, new_num_threads: usize) -> Result<()> {
        if !self.policy.allow_dynamic_scaling {
            return Err(crate::Error::RuntimeError(format!(
                "Resource group '{}' does not allow dynamic scaling",
                self.name
            )));
        }
        self.executor.resize(new_num_threads)
    }

    /// Rebuild the thread pool with new core affinity settings.
    pub fn set_core_affinity(&self, cores: Vec<usize>) -> Result<()> {
        self.executor.set_core_affinity(cores)
    }
}

/// Central scheduler that manages resource groups and routes tasks to devices.
///
/// Supports both local and distributed (file/shared-memory) coordination modes.
/// A singleton instance is accessible via [`scheduler()`].
pub struct TaskScheduler {
    groups: Mutex<HashMap<String, Arc<ResourceGroup>>>,
    mode: OrchestratorMode,
    coordinator: Option<Box<dyn LoadCoordinator>>,
    global_load_cache: Mutex<(Arc<HashMap<DeviceId, usize>>, Instant)>,
    /// Maps failed devices to the time of failure. Devices are considered recovered
    /// after DEVICE_RECOVERY_TIMEOUT has elapsed.
    failed_devices: Mutex<HashMap<DeviceId, Instant>>,
    /// How long to wait for GPU VRAM before giving up.
    /// Defaults from `CV_GPU_WAIT_TIMEOUT_MS` env var or 30 s.
    gpu_wait_timeout: Mutex<Duration>,
    /// Interval for caching global load statistics.
    /// Defaults to 200ms, configurable via `CV_RUNTIME_LOAD_CACHE_MS`.
    load_cache_interval: Duration,
}

/// The level of adaptation to apply in Adaptive mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveLevel {
    /// Basic adaptation: automatic fallback to CPU on GPU error.
    Basic,
    /// Aggressive adaptation: fallback to CPU, plus aggressive group rebalancing/stealing.
    Aggressive,
}

/// The execution mode of the runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Strict mode: fail on error, no automatic fallbacks.
    Strict,
    /// Normal mode: default behavior without dynamic fallbacks, standard orchestrator.
    Normal,
    /// Adaptive mode: automatically fallback to CPU on GPU error and perform adaptive scheduling.
    Adaptive(AdaptiveLevel),
}

use parking_lot::RwLock;

static EXECUTION_MODE: RwLock<ExecutionMode> = parking_lot::const_rwlock(ExecutionMode::Normal);

/// Set the process-wide execution mode.
pub fn set_execution_mode(mode: ExecutionMode) {
    *EXECUTION_MODE.write() = mode;
}

/// Get the current execution mode. Defaults to Normal.
pub fn get_execution_mode() -> ExecutionMode {
    *EXECUTION_MODE.read()
}

/// Reset the execution mode to Normal. For use in tests to ensure isolation.
pub fn reset_execution_mode() {
    *EXECUTION_MODE.write() = ExecutionMode::Normal;
}

/// A handle for executing closures, either on a resource group's thread pool
/// or synchronously on the calling thread.
pub enum RuntimeRunner {
    /// Execute on a resource group's thread pool.
    Group(Arc<ResourceGroup>),
    /// Execute synchronously on the calling thread for the given device.
    Sync(DeviceId),
}

impl RuntimeRunner {
    /// Run a closure on this runner and return its result.
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
    pub fn run_safe<F, R, E>(&self, is_idempotent: bool, f: F) -> std::result::Result<R, E>
    where
        F: Fn() -> std::result::Result<R, E> + Send + Clone,
        R: Send,
        E: From<crate::Error> + Send,
    {
        let f_cloned = f.clone();
        let res = self.run(f_cloned);

        if res.is_err()
            && is_idempotent
            && matches!(get_execution_mode(), ExecutionMode::Adaptive(_))
        {
            // Signal failure to scheduler for future calls
            if let Ok(s) = scheduler() {
                s.report_failure(self.device_id());
            }
            // Fallback to CPU - run on CPU device
            if let Ok(reg) = registry() {
                let cpu_id = reg.default_cpu().id();
                let cpu_runner = RuntimeRunner::Sync(cpu_id);
                warn!(
                    "GPU device {:?} failed, falling back to CPU device {:?}",
                    self.device_id(),
                    cpu_id
                );
                return cpu_runner.run(f);
            }
        }
        res
    }

    /// Return the device ID associated with this runner.
    pub fn device_id(&self) -> DeviceId {
        match self {
            RuntimeRunner::Group(g) => g.device_id(),
            RuntimeRunner::Sync(id) => *id,
        }
    }

    /// Look up this runner's [`DeviceRuntime`] in the global registry.
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

    /// Try to resolve the HAL compute device for this runner.
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

/// Get the best available runtime runner (GPU if available, else CPU).
pub fn best_runner() -> Result<RuntimeRunner> {
    try_best_runner()
}

/// Try to get the best runner; falls back to synchronous CPU if no scheduler is available.
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

/// Get the best GPU runner, waiting for VRAM if needed.
///
/// Waits up to the configured timeout for GPU memory to free up. If the
/// timeout expires, falls back to CPU.  Returns `(runner, is_gpu)` so
/// the caller knows which path was taken.
pub fn best_runner_gpu_wait() -> Result<(RuntimeRunner, bool)> {
    best_runner_gpu_wait_for(WorkloadHint::Default, 1, None)
}

/// Like [`best_runner_gpu_wait`] with explicit hint and timeout.
pub fn best_runner_gpu_wait_for(
    hint: WorkloadHint,
    needed_mb: u32,
    timeout: Option<Duration>,
) -> Result<(RuntimeRunner, bool)> {
    if let Ok(s) = scheduler() {
        if let Ok((group, is_gpu)) = s.best_gpu_with_wait(hint, needed_mb, timeout) {
            return Ok((RuntimeRunner::Group(group), is_gpu));
        }
    }

    if let Ok(reg) = registry() {
        return Ok((RuntimeRunner::Sync(reg.default_cpu().id()), false));
    }

    Err(crate::Error::RuntimeError(
        "Could not initialize even basic device registry".into(),
    ))
}

/// Get the default runtime runner (from the `"default"` group or CPU fallback).
pub fn default_runner() -> Result<RuntimeRunner> {
    try_default_runner()
}

/// Try to get the default runner; falls back to synchronous CPU if no scheduler is available.
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

/// Record a GPU→CPU fallback in the observability layer.
///
/// Call this from any algorithm at the point where it decides to run on CPU
/// instead of GPU so profiling tools can identify silent fallbacks.
///
/// ```ignore
/// if let Ok(ComputeDevice::Gpu(gpu)) = runner.device() {
///     if let Ok(result) = my_gpu_kernel(gpu, ...) { return result; }
///     record_fallback("my_kernel", runner.device_id(), DispatchReason::GpuError);
/// } else {
///     record_fallback("my_kernel", runner.device_id(), DispatchReason::NoGpu);
/// }
/// // ... CPU path ...
/// ```
pub fn record_fallback(
    operation: &str,
    actual_device: DeviceId,
    reason: crate::observe::events::DispatchReason,
) {
    use crate::observe::events::{DispatchBackend, RuntimeEvent};
    use crate::observe::observability;

    observability().publish_event(RuntimeEvent::Dispatch {
        operation: operation.to_string(),
        actual_device,
        intended_backend: DispatchBackend::Gpu,
        actual_backend: DispatchBackend::Cpu,
        reason,
        wait_ms: 0,
        requested_memory_mb: 0,
        affinity_group: 0,
        timestamp: Instant::now(),
    });
}

impl Default for TaskScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskScheduler {
    /// Create a new scheduler, optionally enabling distributed coordination
    /// if `CV_RUNTIME_COORDINATOR` or `CV_RUNTIME_SHM` is set.
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
            } else if let Ok(name) = std::env::var("CV_RUNTIME_SHM") {
                match ShmCoordinator::new(&name, SHM_SIZE) {
                    Ok(c) => {
                        c.start_heartbeat_thread(Duration::from_secs(1));
                        (
                            OrchestratorMode::Distributed {
                                coordinator_path: std::path::PathBuf::from(format!(
                                    "/dev/shm/{}",
                                    name
                                )),
                            },
                            Some(Box::new(c)),
                        )
                    }
                    Err(_) => (OrchestratorMode::Local, None),
                }
            } else {
                (OrchestratorMode::Local, None)
            };

        let scheduler = Self {
            groups: Mutex::new(HashMap::new()),
            mode,
            coordinator,
            global_load_cache: Mutex::new((
                Arc::new(HashMap::new()),
                Instant::now() - Duration::from_secs(3600),
            )),
            failed_devices: Mutex::new(HashMap::new()),
            gpu_wait_timeout: Mutex::new(gpu_wait_timeout()),
            load_cache_interval: load_cache_interval(),
        };

        // Auto-initialize GPU devices in the coordinator so memory budgets work.
        // This runs once at startup — zero cost on the compute path.
        scheduler.auto_init_devices();

        scheduler
    }

    /// Discover GPU devices from the registry and register their memory capacity
    /// with the coordinator. Only the first process to initialize a device wins
    /// (CAS inside init_device), so this is safe to call from every process.
    fn auto_init_devices(&self) {
        let coord = match self.coordinator.as_ref() {
            Some(c) => c,
            None => return,
        };

        let reg = match registry() {
            Ok(r) => r,
            Err(_) => return,
        };

        let mut device_idx: u8 = 0;
        for device in reg.all_devices() {
            if device.backend() == BackendType::Cpu {
                continue;
            }
            let mem_mb = device.estimated_memory_mb();
            if mem_mb > 0 && device_idx < 8 {
                // Use the coordinator's init_device — CAS ensures only first writer wins
                let _ = coord.init_device(device_idx, mem_mb);
                device_idx += 1;
            }
        }
    }

    /// Return the current orchestration mode (local or distributed).
    pub fn mode(&self) -> &OrchestratorMode {
        &self.mode
    }

    /// Record a device failure so subsequent scheduling avoids it temporarily.
    pub fn report_failure(&self, device_id: DeviceId) {
        let mut failed = self.failed_devices.lock();
        failed.insert(device_id, Instant::now());
    }

    /// Check whether a device is healthy (has not failed recently).
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

    /// Create a new resource group on the default CPU device.
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

    /// Create a new resource group bound to a specific device.
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
                .add_executor(group.executor.clone());
        }

        Ok(group)
    }

    /// Remove and return a resource group by name, if it exists.
    pub fn remove_group(&self, name: &str) -> Result<Option<Arc<ResourceGroup>>> {
        let mut groups = self.groups.lock();
        if let Some(group) = groups.remove(name) {
            Ok(Some(group))
        } else {
            Ok(None)
        }
    }

    /// Look up a resource group by name.
    pub fn get_group(&self, name: &str) -> Result<Option<Arc<ResourceGroup>>> {
        let groups = self.groups.lock();
        Ok(groups.get(name).cloned())
    }

    fn get_global_load(&self) -> Arc<HashMap<DeviceId, usize>> {
        let mut cache = self.global_load_cache.lock();
        if cache.1.elapsed() > self.load_cache_interval {
            if let Some(ref coord) = self.coordinator {
                let local_load = self.get_local_load();
                if let Err(_e) = coord.update_load(&local_load) {
                    #[cfg(feature = "tracing")]
                    tracing::warn!("Failed to update coordinator load: {}", e);
                }

                if let Ok(global) = coord.get_global_load() {
                    cache.0 = Arc::new(global);
                }
            }
            cache.1 = Instant::now();
        }
        cache.0.clone() // Arc clone = atomic pointer bump, no heap alloc
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
    ///
    /// If an affinity group is active, it prefers devices where peers are already
    /// scheduled, provided the device load is within a reasonable threshold.
    pub fn get_best_group(
        &self,
        backend_type: BackendType,
        hint: WorkloadHint,
    ) -> Result<Option<Arc<ResourceGroup>>> {
        let global_load = self.get_global_load();
        let groups = self.groups.lock();

        // Affinity awareness: check if coordinator has a preferred device for our group
        let affinity_device = self
            .coordinator
            .as_ref()
            .and_then(|c| c.best_device_for_group(1));

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

            // Use cached backend — no registry mutex hit.
            let group_backend = group.backend();

            let matches = match hint {
                WorkloadHint::Require(req) => group_backend == req,
                WorkloadHint::Prefer(pref) => {
                    // Match any GPU if pref is a GPU type, otherwise exact match
                    if pref != BackendType::Cpu && group_backend != BackendType::Cpu {
                        true
                    } else {
                        group_backend == pref
                    }
                }
                _ => match (backend_type, group_backend) {
                    (BackendType::Cpu, BackendType::Cpu) => true,
                    // Any GPU backend matches a GPU request for now
                    (t, b) if t != BackendType::Cpu && b != BackendType::Cpu => true,
                    _ => false,
                },
            };

            if matches {
                let priority = group.policy.priority;

                // Adjust selection logic based on hint
                if hint == WorkloadHint::Latency
                    && priority < TaskPriority::Normal
                    && get_execution_mode() != ExecutionMode::Adaptive(AdaptiveLevel::Aggressive)
                {
                    continue; // Skip low priority for latency sensitive unless in aggressive mode
                }

                // Load calculation: Local device load + remote load for this device
                let local_device_load = local_load_by_device.get(&device_id).copied().unwrap_or(0);
                let remote_load = global_load
                    .get(&device_id)
                    .copied()
                    .unwrap_or(0)
                    .saturating_sub(local_device_load);
                let total_device_load = group.load() + remote_load;

                // Affinity preference: If this is our affinity device, treat it as having less load
                // (threshold of 2 tasks) to encourage co-placement.
                let effective_load = if Some((device_id.0 & 0xFF) as u8) == affinity_device {
                    total_device_load.saturating_sub(2)
                } else {
                    total_device_load
                };

                // Boost priority if it's our preferred backend
                let final_priority = priority;
                if let WorkloadHint::Prefer(pref) = hint {
                    if group_backend == pref {
                        // Conceptual priority boost for exact preferred backend match
                        // (Internal only, doesn't change policy)
                        if final_priority < TaskPriority::Critical {
                            // We can't actually increment the enum, but we can influence the choice.
                            // For simplicity, we just use the min_load check below.
                        }
                    }
                }

                if final_priority > max_priority {
                    max_priority = final_priority;
                    min_load = effective_load;
                    best_group = Some(group.clone());
                } else if final_priority == max_priority && effective_load < min_load {
                    min_load = effective_load;
                    best_group = Some(group.clone());
                }
            }
        }

        Ok(best_group)
    }

    /// Return the `"default"` resource group.
    pub fn get_default_group(&self) -> Result<Arc<ResourceGroup>> {
        self.get_group("default")?.ok_or_else(|| {
            crate::Error::RuntimeError("Default resource group not found".to_string())
        })
    }

    /// Return the best GPU group, falling back to the default CPU group.
    pub fn best_gpu_or_cpu(&self) -> Result<Arc<ResourceGroup>> {
        self.best_gpu_or_cpu_for(WorkloadHint::Default)
    }

    /// Like [`best_gpu_or_cpu`](Self::best_gpu_or_cpu) but with a workload hint for scheduling.
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

    /// Get the best GPU group, waiting for VRAM if the device is currently full.
    /// Falls back to CPU only after the timeout expires — GPU work stays on GPU
    /// as long as possible.
    ///
    /// Flow: pick GPU → VRAM available? run on GPU : wait up to timeout →
    ///       VRAM freed? run on GPU : fall back to CPU.
    ///
    /// The timeout is read from (in priority order):
    /// 1. The explicit `timeout` parameter (if `Some`)
    /// 2. `set_gpu_wait_timeout()` (per-process override)
    /// 3. `CV_GPU_WAIT_TIMEOUT_MS` environment variable (system-wide)
    /// 4. 30 second default
    ///
    /// Every decision is recorded as a [`RuntimeEvent::Dispatch`] in the
    /// observability layer so profiling tools can see fallbacks.
    ///
    /// Returns `(group, true)` when running on GPU, `(group, false)` on CPU fallback.
    pub fn best_gpu_with_wait(
        &self,
        hint: WorkloadHint,
        needed_mb: u32,
        timeout: Option<Duration>,
    ) -> Result<(Arc<ResourceGroup>, bool)> {
        use crate::observe::events::{DispatchBackend, DispatchReason};
        use crate::observe::{observability, RuntimeEvent};

        let gpu_group = self
            .get_best_group(BackendType::WebGPU, hint)?
            .or(self.get_best_group(BackendType::Vulkan, hint)?);

        let gpu_group = match gpu_group {
            Some(g) => g,
            None => {
                // No GPU groups at all → CPU
                let cpu = self.get_default_group()?;
                observability().publish_event(RuntimeEvent::Dispatch {
                    operation: String::new(),
                    actual_device: cpu.device_id(),
                    intended_backend: DispatchBackend::Gpu,
                    actual_backend: DispatchBackend::Cpu,
                    reason: DispatchReason::NoGpu,
                    wait_ms: 0,
                    requested_memory_mb: needed_mb,
                    affinity_group: 0,
                    timestamp: Instant::now(),
                });
                return Ok((cpu, false));
            }
        };

        // If there's a coordinator, check VRAM budget
        if let Some(ref coord) = self.coordinator {
            let dev_idx = (gpu_group.device_id().0 & 0xFF) as u8;
            let usage = coord.device_memory_usage();
            // Use device index as affinity group identifier
            let affinity_group: u32 = dev_idx as u32;

            let is_over_budget = usage.iter().any(|&(idx, used, total)| {
                idx == dev_idx && total > 0 && (used + needed_mb) > total
            });

            if is_over_budget {
                let wait = timeout.unwrap_or_else(|| *self.gpu_wait_timeout.lock());

                if wait.is_zero() {
                    let cpu = self.get_default_group()?;
                    observability().publish_event(RuntimeEvent::Dispatch {
                        operation: String::new(),
                        actual_device: cpu.device_id(),
                        intended_backend: DispatchBackend::Gpu,
                        actual_backend: DispatchBackend::Cpu,
                        reason: DispatchReason::VramTimeout,
                        wait_ms: 0,
                        requested_memory_mb: needed_mb,
                        affinity_group,
                        timestamp: Instant::now(),
                    });
                    return Ok((cpu, false));
                }

                let wait_start = Instant::now();
                match coord.wait_for_device_memory(dev_idx, needed_mb, wait) {
                    Ok(()) => {
                        let waited = wait_start.elapsed().as_millis() as u64;
                        observability().publish_event(RuntimeEvent::Dispatch {
                            operation: String::new(),
                            actual_device: gpu_group.device_id(),
                            intended_backend: DispatchBackend::Gpu,
                            actual_backend: DispatchBackend::Gpu,
                            reason: DispatchReason::VramFreedAfterWait,
                            wait_ms: waited,
                            requested_memory_mb: needed_mb,
                            affinity_group,
                            timestamp: Instant::now(),
                        });
                        return Ok((gpu_group, true));
                    }
                    Err(_) => {
                        let waited = wait_start.elapsed().as_millis() as u64;
                        let cpu = self.get_default_group()?;
                        observability().publish_event(RuntimeEvent::Dispatch {
                            operation: String::new(),
                            actual_device: cpu.device_id(),
                            intended_backend: DispatchBackend::Gpu,
                            actual_backend: DispatchBackend::Cpu,
                            reason: DispatchReason::VramTimeout,
                            wait_ms: waited,
                            requested_memory_mb: needed_mb,
                            affinity_group,
                            timestamp: Instant::now(),
                        });
                        return Ok((cpu, false));
                    }
                }
            }
        }

        // GPU is available (no coordinator or not over budget)
        let affinity_group = self.coordinator.as_ref().map(|_| 0).unwrap_or(0);
        observability().publish_event(RuntimeEvent::Dispatch {
            operation: String::new(),
            actual_device: gpu_group.device_id(),
            intended_backend: DispatchBackend::Gpu,
            actual_backend: DispatchBackend::Gpu,
            reason: DispatchReason::GpuAvailable,
            wait_ms: 0,
            requested_memory_mb: needed_mb,
            affinity_group,
            timestamp: Instant::now(),
        });
        Ok((gpu_group, true))
    }

    /// Set the process-wide GPU VRAM wait timeout.
    ///
    /// Overrides the `CV_GPU_WAIT_TIMEOUT_MS` env var for this process.
    /// Pass `Duration::ZERO` to disable waiting (fail immediately on over-budget).
    pub fn set_gpu_wait_timeout(&self, timeout: Duration) {
        *self.gpu_wait_timeout.lock() = timeout;
    }

    /// Get the current GPU wait timeout.
    pub fn gpu_wait_timeout(&self) -> Duration {
        *self.gpu_wait_timeout.lock()
    }

    /// Wait for VRAM on a specific device, using the configured timeout.
    ///
    /// Blocks until the device has at least `needed_mb` free or the timeout
    /// expires. Returns `Err` on timeout.
    pub fn wait_for_gpu(&self, device_idx: u8, needed_mb: u32) -> Result<()> {
        self.wait_for_gpu_with_timeout(device_idx, needed_mb, None)
    }

    /// Wait for VRAM with an explicit timeout override.
    pub fn wait_for_gpu_with_timeout(
        &self,
        device_idx: u8,
        needed_mb: u32,
        timeout: Option<Duration>,
    ) -> Result<()> {
        let coord = self.coordinator.as_ref().ok_or_else(|| {
            crate::Error::RuntimeError("No coordinator available for VRAM waiting".into())
        })?;
        let wait = timeout.unwrap_or_else(|| *self.gpu_wait_timeout.lock());
        coord
            .wait_for_device_memory(device_idx, needed_mb, wait)
            .map_err(|e| crate::Error::RuntimeError(format!("GPU VRAM wait failed: {}", e)))
    }

    /// Submit a task to the named resource group for asynchronous execution.
    pub fn submit<F>(&self, group_name: &str, f: F) -> Result<()>
    where
        F: FnOnce() + Send + 'static,
    {
        if let Some(group) = self.get_group(group_name)? {
            group.spawn(f)?;
            Ok(())
        } else {
            Err(crate::Error::RuntimeError(format!(
                "Resource group '{}' not found",
                group_name
            )))
        }
    }

    // --- Coordinator delegation ---

    /// Query per-device memory usage from the coordinator, if available.
    pub fn coordinator_device_usage(&self) -> Option<Vec<(u8, u32, u32)>> {
        self.coordinator.as_ref().map(|c| c.device_memory_usage())
    }

    /// Exposes init_device directly for testing and explicit Python configuration.
    pub fn mock_init_device(&self, device_idx: u8, total_mb: u32) -> std::io::Result<()> {
        if let Some(ref coord) = self.coordinator {
            coord.init_device(device_idx, total_mb)
        } else {
            Err(std::io::Error::other("No coordinator"))
        }
    }

    /// Join an affinity group via the coordinator.
    pub fn join_affinity_group(&self, group_id: u32) -> Result<()> {
        match self.coordinator.as_ref() {
            Some(c) => c
                .join_group(group_id)
                .map_err(|e| crate::Error::RuntimeError(format!("Failed to join group: {}", e))),
            None => Err(crate::Error::RuntimeError(
                "No coordinator available".into(),
            )),
        }
    }

    /// Reserve a device via the coordinator.
    pub fn reserve_device(&self, device_idx: u8, memory_mb: u32) -> Result<()> {
        match self.coordinator.as_ref() {
            Some(c) => c.reserve_device(device_idx, memory_mb, 0).map_err(|e| {
                crate::Error::RuntimeError(format!("Failed to reserve device: {}", e))
            }),
            None => Err(crate::Error::RuntimeError(
                "No coordinator available".into(),
            )),
        }
    }

    /// Release a device reservation via the coordinator.
    pub fn release_device(&self, device_idx: u8) -> Result<()> {
        match self.coordinator.as_ref() {
            Some(c) => c.release_device(device_idx).map_err(|e| {
                crate::Error::RuntimeError(format!("Failed to release device: {}", e))
            }),
            None => Err(crate::Error::RuntimeError(
                "No coordinator available".into(),
            )),
        }
    }

    /// Find the best device considering affinity group via the coordinator.
    pub fn best_device_for_group(&self, needed_mb: u32) -> Option<u8> {
        self.coordinator
            .as_ref()
            .and_then(|c| c.best_device_for_group(needed_mb))
    }

    /// Re-scan for new or recovered devices and update the registry.
    pub fn refresh_devices(&self) -> Result<()> {
        registry()?.refresh()?;
        // After refreshing, we might need to re-init devices in the coordinator
        self.auto_init_devices();
        Ok(())
    }
}

static GLOBAL_SCHEDULER: OnceLock<Result<TaskScheduler>> = OnceLock::new();

/// Return the global [`TaskScheduler`] singleton, creating it on first call.
///
/// Initializes a `"default"` resource group with one thread per logical CPU core.
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
        std::thread::sleep(Duration::from_millis(50));
        assert!(g1.load() >= 1);

        if let Ok(Some(best)) = s.get_best_group(BackendType::Cpu, WorkloadHint::Default) {
            assert_eq!(best.name, "g2");
        }
    }

    #[test]
    fn test_execution_mode_default() {
        reset_execution_mode();
        assert_eq!(get_execution_mode(), ExecutionMode::Normal);
    }

    #[test]
    fn test_execution_mode_strict() {
        reset_execution_mode();
        set_execution_mode(ExecutionMode::Strict);
        assert_eq!(get_execution_mode(), ExecutionMode::Strict);
    }

    #[test]
    fn test_execution_mode_normal() {
        reset_execution_mode();
        set_execution_mode(ExecutionMode::Normal);
        assert_eq!(get_execution_mode(), ExecutionMode::Normal);
    }

    #[test]
    fn test_execution_mode_adaptive_basic() {
        reset_execution_mode();
        set_execution_mode(ExecutionMode::Adaptive(AdaptiveLevel::Basic));
        match get_execution_mode() {
            ExecutionMode::Adaptive(level) => assert_eq!(level, AdaptiveLevel::Basic),
            _ => panic!("Expected Adaptive(Basic)"),
        }
    }

    #[test]
    fn test_execution_mode_adaptive_aggressive() {
        reset_execution_mode();
        set_execution_mode(ExecutionMode::Adaptive(AdaptiveLevel::Aggressive));
        match get_execution_mode() {
            ExecutionMode::Adaptive(level) => assert_eq!(level, AdaptiveLevel::Aggressive),
            _ => panic!("Expected Adaptive(Aggressive)"),
        }
    }

    #[test]
    fn test_execution_mode_reset() {
        set_execution_mode(ExecutionMode::Strict);
        assert_eq!(get_execution_mode(), ExecutionMode::Strict);
        reset_execution_mode();
        assert_eq!(get_execution_mode(), ExecutionMode::Normal);
    }

    #[test]
    fn test_scheduler_new_instance() {
        let s1 = TaskScheduler::new();
        let s2 = TaskScheduler::new();
        let policy = GroupPolicy::default();

        s1.create_group("test", 2, None, policy.clone()).unwrap();

        assert!(s1.get_group("test").unwrap().is_some());
        assert!(s2.get_group("test").unwrap().is_none());
    }

    #[test]
    fn test_group_remove() {
        let s = TaskScheduler::new();
        let policy = GroupPolicy::default();

        s.create_group("test", 2, None, policy).unwrap();
        assert!(s.get_group("test").unwrap().is_some());

        let removed = s.remove_group("test").unwrap();
        assert!(removed.is_some());
        assert!(s.get_group("test").unwrap().is_none());
    }

    #[test]
    fn test_group_remove_nonexistent() {
        let s = TaskScheduler::new();
        let removed = s.remove_group("nonexistent").unwrap();
        assert!(removed.is_none());
    }

    #[test]
    fn test_group_get() {
        let s = TaskScheduler::new();
        let policy = GroupPolicy::default();

        let _group = s.create_group("test", 2, None, policy).unwrap();
        let retrieved = s.get_group("test").unwrap();

        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "test");
    }

    #[test]
    fn test_group_get_nonexistent() {
        let s = TaskScheduler::new();
        let retrieved = s.get_group("nonexistent").unwrap();
        assert!(retrieved.is_none());
    }

    #[test]
    fn test_device_health_check() {
        let s = TaskScheduler::new();
        let cpu_id = registry().unwrap().default_cpu().id();

        assert!(s.is_device_healthy(cpu_id));
    }

    #[test]
    fn test_task_priority_ordering() {
        assert!(TaskPriority::Critical > TaskPriority::High);
        assert!(TaskPriority::High > TaskPriority::Normal);
        assert!(TaskPriority::Normal > TaskPriority::Low);
        assert!(TaskPriority::Low > TaskPriority::Background);
    }

    #[test]
    fn test_runtime_runner_device_id() {
        let runner = RuntimeRunner::Sync(DeviceId(42));
        assert_eq!(runner.device_id(), DeviceId(42));
    }

    #[test]
    fn test_runtime_runner_run_sync() {
        let runner = RuntimeRunner::Sync(DeviceId(0));
        let result = runner.run(|| 42);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_runtime_runner_run_safe_success() {
        let runner = RuntimeRunner::Sync(DeviceId(0));
        let result: std::result::Result<i32, crate::Error> = runner.run_safe(true, || Ok(42));
        assert_eq!(result.unwrap(), 42);
    }
}
