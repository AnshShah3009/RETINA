use cv_runtime::device_registry::registry;
use cv_runtime::orchestrator::{scheduler, AdaptiveLevel, ExecutionMode, WorkloadHint};
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone, Copy)]
pub enum PyWorkloadHint {
    Latency,
    Throughput,
    PowerSave,
    Default,
}

impl From<PyWorkloadHint> for WorkloadHint {
    fn from(hint: PyWorkloadHint) -> Self {
        match hint {
            PyWorkloadHint::Latency => WorkloadHint::Latency,
            PyWorkloadHint::Throughput => WorkloadHint::Throughput,
            PyWorkloadHint::PowerSave => WorkloadHint::PowerSave,
            PyWorkloadHint::Default => WorkloadHint::Default,
        }
    }
}

#[pyclass]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyExecutionMode {
    Strict,
    Normal,
    AdaptiveBasic,
    AdaptiveAggressive,
}

impl From<PyExecutionMode> for ExecutionMode {
    fn from(mode: PyExecutionMode) -> Self {
        match mode {
            PyExecutionMode::Strict => ExecutionMode::Strict,
            PyExecutionMode::Normal => ExecutionMode::Normal,
            PyExecutionMode::AdaptiveBasic => ExecutionMode::Adaptive(AdaptiveLevel::Basic),
            PyExecutionMode::AdaptiveAggressive => {
                ExecutionMode::Adaptive(AdaptiveLevel::Aggressive)
            }
        }
    }
}

/// Per-device memory information exposed to Python.
#[pyclass]
#[derive(Clone)]
pub struct PyDeviceInfo {
    #[pyo3(get)]
    pub device_idx: u8,
    #[pyo3(get)]
    pub total_mb: u32,
    #[pyo3(get)]
    pub used_mb: u32,
    #[pyo3(get)]
    pub owner_count: u32,
}

#[pymethods]
impl PyDeviceInfo {
    pub fn free_mb(&self) -> u32 {
        self.total_mb.saturating_sub(self.used_mb)
    }

    fn __repr__(&self) -> String {
        format!(
            "DeviceInfo(idx={}, used={}/{}MB, owners={})",
            self.device_idx, self.used_mb, self.total_mb, self.owner_count
        )
    }
}

/// Handle for an affinity group membership.
#[pyclass]
pub struct PyAffinityGroup {
    #[pyo3(get)]
    pub group_id: u32,
}

#[pymethods]
impl PyAffinityGroup {
    /// Leave this affinity group.
    pub fn leave(&self) -> PyResult<()> {
        let s = scheduler()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        s.join_affinity_group(0)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!("AffinityGroup(id={})", self.group_id)
    }
}

#[pyclass]
pub struct PyRuntime;

#[pymethods]
impl PyRuntime {
    /// Get the aggregate per-device load from all coordinated processes.
    ///
    /// Returns:
    ///     List of (device_id, load) tuples.
    #[staticmethod]
    pub fn get_global_load() -> PyResult<Vec<(u32, usize)>> {
        let s = scheduler()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        // Return coordinator device usage if available
        if let Some(usage) = s.coordinator_device_usage() {
            Ok(usage
                .into_iter()
                .map(|(idx, used, _total)| (idx as u32, used as usize))
                .collect())
        } else {
            Ok(vec![])
        }
    }

    /// Get the number of compute devices registered in the runtime.
    ///
    /// Returns:
    ///     Number of devices (CPU + GPU).
    #[staticmethod]
    pub fn get_num_devices() -> usize {
        match registry() {
            Ok(reg) => reg.all_devices().len(),
            Err(_) => 1,
        }
    }

    /// Get detailed info for all devices tracked by the coordinator.
    ///
    /// Returns:
    ///     List of DeviceInfo objects with memory usage details.
    #[staticmethod]
    pub fn get_device_info() -> PyResult<Vec<PyDeviceInfo>> {
        let s = scheduler()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        match s.coordinator_device_usage() {
            Some(usage) => Ok(usage
                .into_iter()
                .map(|(idx, used, total)| PyDeviceInfo {
                    device_idx: idx,
                    total_mb: total,
                    used_mb: used,
                    owner_count: 0, // not tracked at coordinator level
                })
                .collect()),
            None => Ok(vec![]),
        }
    }

    /// Reserve memory on a specific device.
    ///
    /// Args:
    ///     device_idx: Device index (0-7).
    ///     memory_mb: Memory to reserve in megabytes.
    ///
    /// Raises:
    ///     RuntimeError: If reservation fails (over budget or no coordinator).
    #[staticmethod]
    pub fn reserve_device(device_idx: u8, memory_mb: u32) -> PyResult<()> {
        let s = scheduler()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        s.reserve_device(device_idx, memory_mb)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Release a device reservation.
    ///
    /// Args:
    ///     device_idx: Device index to release.
    #[staticmethod]
    pub fn release_device(device_idx: u8) -> PyResult<()> {
        let s = scheduler()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        s.release_device(device_idx)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Join an affinity group for co-placement with peer processes.
    ///
    /// Args:
    ///     group_id: Non-zero group identifier.
    ///
    /// Returns:
    ///     AffinityGroup handle that can be used to leave the group.
    #[staticmethod]
    pub fn join_group(group_id: u32) -> PyResult<PyAffinityGroup> {
        let s = scheduler()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        s.join_affinity_group(group_id)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyAffinityGroup { group_id })
    }

    /// Find the best device for this process considering affinity peers.
    ///
    /// Args:
    ///     needed_mb: Minimum free memory required in megabytes.
    ///
    /// Returns:
    ///     Device index, or -1 if no suitable device found.
    #[staticmethod]
    pub fn best_device(needed_mb: u32) -> i32 {
        match scheduler() {
            Ok(s) => s.best_device_for_group(needed_mb).map_or(-1, |d| d as i32),
            Err(_) => -1,
        }
    }

    /// Wait for GPU VRAM to become available on a device.
    ///
    /// Blocks until the device has at least `needed_mb` free or the timeout
    /// expires. If no timeout is given, uses the process-wide default
    /// (set via `set_gpu_wait_timeout` or `CV_GPU_WAIT_TIMEOUT_MS`, default 30s).
    ///
    /// Args:
    ///     device_idx: Device index (0-7).
    ///     needed_mb: Minimum free memory required in megabytes.
    ///     timeout_ms: Optional timeout in milliseconds. None uses the default.
    ///
    /// Raises:
    ///     RuntimeError: If the wait times out.
    #[staticmethod]
    #[pyo3(signature = (device_idx, needed_mb, timeout_ms=None))]
    pub fn wait_for_gpu(device_idx: u8, needed_mb: u32, timeout_ms: Option<u64>) -> PyResult<()> {
        let s = scheduler()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let timeout = timeout_ms.map(std::time::Duration::from_millis);
        s.wait_for_gpu_with_timeout(device_idx, needed_mb, timeout)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Set the process-wide GPU VRAM wait timeout.
    ///
    /// Overrides `CV_GPU_WAIT_TIMEOUT_MS` for this process. Pass 0 to disable
    /// waiting (fail immediately when GPU memory is full).
    ///
    /// Args:
    ///     timeout_ms: Timeout in milliseconds.
    #[staticmethod]
    pub fn set_gpu_wait_timeout(timeout_ms: u64) -> PyResult<()> {
        let s = scheduler()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        s.set_gpu_wait_timeout(std::time::Duration::from_millis(timeout_ms));
        Ok(())
    }

    /// Get the current GPU wait timeout in milliseconds.
    #[staticmethod]
    pub fn get_gpu_wait_timeout() -> PyResult<u64> {
        let s = scheduler()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(s.gpu_wait_timeout().as_millis() as u64)
    }

    /// Get recent dispatch events showing GPU/CPU routing decisions.
    ///
    /// Each entry is a dict with: operation, intended, actual, reason, wait_ms.
    /// Useful for profiling to find silent CPU fallbacks.
    ///
    /// Args:
    ///     count: Max number of recent events to return (default 50).
    ///
    /// Returns:
    ///     List of dicts describing each dispatch decision.
    #[staticmethod]
    #[pyo3(signature = (count=50))]
    pub fn get_dispatch_log(count: usize) -> Vec<std::collections::HashMap<String, String>> {
        use cv_runtime::observe::events::RuntimeEvent;

        let layer = cv_runtime::observability();
        let events = layer.get_recent_events(count * 2); // over-fetch, then filter

        events
            .into_iter()
            .filter_map(|e| {
                if let RuntimeEvent::Dispatch {
                    operation,
                    intended_backend,
                    actual_backend,
                    reason,
                    wait_ms,
                    ..
                } = e
                {
                    let mut m = std::collections::HashMap::new();
                    m.insert("operation".into(), operation);
                    m.insert("intended".into(), format!("{:?}", intended_backend));
                    m.insert("actual".into(), format!("{:?}", actual_backend));
                    m.insert("reason".into(), format!("{:?}", reason));
                    m.insert("wait_ms".into(), wait_ms.to_string());
                    m.insert(
                        "fallback".into(),
                        (intended_backend != actual_backend).to_string(),
                    );
                    Some(m)
                } else {
                    None
                }
            })
            .take(count)
            .collect()
    }

    /// Manually initialize a device in the coordinator with a specific memory limit.
    /// Useful for testing coordination when no real GPU is present.
    #[staticmethod]
    pub fn mock_init_device(device_idx: u8, total_mb: u32) -> PyResult<()> {
        let s = scheduler()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        s.mock_init_device(device_idx, total_mb)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(())
    }

    /// Set the process-wide execution mode.
    #[staticmethod]
    pub fn set_execution_mode(mode: PyExecutionMode) {
        cv_runtime::orchestrator::set_execution_mode(mode.into());
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWorkloadHint>()?;
    m.add_class::<PyExecutionMode>()?;
    m.add_class::<PyRuntime>()?;
    m.add_class::<PyDeviceInfo>()?;
    m.add_class::<PyAffinityGroup>()?;
    Ok(())
}
