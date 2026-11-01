//! A runtime CPU runner must resolve to a HAL CPU compute device.
//!
//! `ResourceGroup::device()` / `RuntimeRunner::device()` are the only bridge
//! between the runtime's device selection and the HAL kernels an algorithm
//! dispatches. `cv-runtime`'s own `BEHAVIOUR.md` documents the idiom as
//!
//! ```ignore
//! if let Ok(ComputeDevice::Gpu(gpu)) = runner.device() {
//!     // GPU kernel
//! } else {
//!     record_fallback("my_kernel", runner.device_id(), DispatchReason::NoGpu);
//! }
//! // CPU path
//! ```
//!
//! and `cv-registration`'s `registration_icp_point_to_plane_ctx` takes a
//! `&ComputeDevice` and early-returns to its CPU ICP implementation for
//! `ComputeDevice::Cpu`. Both assume a CPU runner yields `ComputeDevice::Cpu`.
//!
//! It does not. The runtime registers its CPU device in `DeviceRegistry` under
//! the id of a `CpuBackend` it constructs itself, while `try_device()` resolves
//! that id through `cv_hal::compute::get_device_by_id`, which only knows the
//! *HAL's* lazily-initialised `CPU_CONTEXT` - a different `CpuBackend`
//! instance, with a different id from the same counter. So every CPU runner
//! (`best_runner()` / `default_runner()` on a CPU-only machine, and the CPU
//! fallback arm of `best_gpu_with_wait` on any machine) reports
//!
//! ```text
//! Err(RuntimeError("Could not find compute device DeviceId(0):
//!     Device error: Device DeviceId(0) not found in global contexts"))
//! ```
//!
//! The CPU path is therefore unreachable: an algorithm that asks a CPU runner
//! for its device gets an error instead of the CPU kernel it should run.

use cv_hal::compute::ComputeDevice;
use cv_hal::BackendType;
use cv_runtime::orchestrator::{GroupPolicy, RuntimeRunner, TaskScheduler};

/// A scheduler whose only device is the default CPU group - what a CPU-only
/// host has, and what the CPU-fallback arm of `best_gpu_with_wait` returns.
fn cpu_only_scheduler() -> TaskScheduler {
    let s = TaskScheduler::new();
    s.create_group("default", 1, None, GroupPolicy::default())
        .expect("the default CPU group must be creatable");
    s
}

/// CONTROL: the group really is a CPU group bound to the registry's CPU device.
/// Without this, the assertions below could pass for the wrong reason (a group
/// bound to a GPU id would make `device()` resolve through the GPU path).
fn assert_is_the_cpu_group(group: &cv_runtime::orchestrator::ResourceGroup) {
    assert_eq!(
        group.backend(),
        BackendType::Cpu,
        "premise: 'default' must be a CPU group"
    );
    assert_eq!(
        group.device_id(),
        cv_runtime::device_registry::registry()
            .expect("registry")
            .default_cpu()
            .id(),
        "premise: 'default' must be bound to the registry's CPU device"
    );
}

/// The resource group a CPU dispatch happens on must hand back the CPU
/// `ComputeDevice`, not an error.
#[test]
fn a_cpu_resource_group_resolves_its_compute_device() {
    let s = cpu_only_scheduler();
    let group = s.get_group("default").expect("lookup").expect("present");
    assert_is_the_cpu_group(&group);

    match group.device() {
        Ok(device) => {
            assert!(
                matches!(device, ComputeDevice::Cpu(_)),
                "a CPU resource group resolved to a non-CPU compute device: {:?}",
                device.backend_type()
            );
            assert_eq!(
                device.device_id(),
                group.device_id(),
                "the resolved device must be the one the group is bound to"
            );
        }
        Err(e) => panic!(
            "a CPU resource group could not resolve a compute device: {e}. \
             The CPU kernel path is unreachable through the runtime."
        ),
    }
}

/// Same contract through the `RuntimeRunner` the scheduler hands out, including
/// the synchronous CPU runner used when no scheduler is available.
#[test]
fn a_cpu_runtime_runner_resolves_its_compute_device() {
    let s = cpu_only_scheduler();
    let group = s.get_group("default").expect("lookup").expect("present");
    let runner = RuntimeRunner::Group(group);

    match runner.try_device() {
        Ok(device) => {
            assert!(matches!(device, ComputeDevice::Cpu(_)));
            assert_eq!(device.device_id(), runner.device_id());
        }
        Err(e) => panic!("a CPU runner could not resolve a compute device: {e}"),
    }

    // The GPU-wait entry point falls back to the default CPU group with
    // `is_gpu == false`. A caller that trusts that flag takes the CPU branch and
    // then asks the runner for its device.
    let (fallback_group, is_gpu) = s
        .best_gpu_with_wait(
            cv_runtime::orchestrator::WorkloadHint::Require(BackendType::Cpu),
            0,
            Some(std::time::Duration::ZERO),
        )
        .expect("a CPU-only scheduler must still answer");
    assert!(!is_gpu, "premise: this is the CPU fallback path");
    let fallback = RuntimeRunner::Group(fallback_group);
    assert!(
        matches!(fallback.try_device(), Ok(ComputeDevice::Cpu(_))),
        "the CPU fallback runner could not resolve a compute device: {:?}",
        fallback.try_device().map(|d| d.backend_type())
    );

    // And the synchronous CPU runner, which `try_best_runner` uses when the
    // scheduler is unavailable.
    let sync = RuntimeRunner::Sync(runner.device_id());
    assert!(
        matches!(sync.try_device(), Ok(ComputeDevice::Cpu(_))),
        "the synchronous CPU runner could not resolve a compute device"
    );
}
