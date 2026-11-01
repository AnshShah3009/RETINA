//! The device's buffer-size limits come from the adapter, not a fixed cap.
//!
//! `GpuContext` used to request `max_buffer_size` and
//! `max_storage_buffer_binding_size` of at most 256 MiB, whatever the adapter
//! offered, so a 4K float image stack or a 20M-point cloud failed validation
//! on a 12 GiB card. wgpu 29 made the binding limit `u64` and reports the
//! native backends' real limits, so the device now asks for what the adapter
//! has, while every other limit stays at the downlevel defaults the shader
//! corpus is checked against, and nothing is ever requested beyond the
//! adapter (`request_device` refuses that outright).

use cv_core::storage::Storage;
use cv_core::{CpuTensor, Tensor, TensorShape};
use cv_hal::context::ComputeContext;
use cv_hal::cpu::CpuBackend;
use cv_hal::gpu::{GpuContext, requested_device_limits};
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};

const MIB: u64 = 1024 * 1024;

/// A desktop-class adapter: 4 GiB buffers, 2 GiB storage bindings.
fn big_adapter() -> wgpu::Limits {
    wgpu::Limits {
        max_buffer_size: 4096 * MIB - 1,
        max_storage_buffer_binding_size: 2048 * MIB - 4,
        ..wgpu::Limits::default()
    }
}

#[test]
fn buffer_limits_are_the_adapters_own() {
    let adapter = big_adapter();
    let req = requested_device_limits(&adapter);
    assert_eq!(req.max_buffer_size, adapter.max_buffer_size);
    assert_eq!(
        req.max_storage_buffer_binding_size,
        adapter.max_storage_buffer_binding_size
    );
}

#[test]
fn other_limits_stay_at_the_downlevel_defaults() {
    // The shader corpus is validated against downlevel_defaults (see
    // shader_bindings_limits.rs); a device that allowed more would let a
    // non-portable shader through on this machine and fail elsewhere.
    let req = requested_device_limits(&big_adapter());
    let down = wgpu::Limits::downlevel_defaults();
    assert_eq!(
        req.max_storage_buffers_per_shader_stage,
        down.max_storage_buffers_per_shader_stage
    );
    assert_eq!(
        req.max_compute_workgroups_per_dimension,
        down.max_compute_workgroups_per_dimension
    );
    assert_eq!(
        req.max_compute_invocations_per_workgroup,
        down.max_compute_invocations_per_workgroup
    );
}

#[test]
fn nothing_is_requested_beyond_the_adapter() {
    for (label, adapter) in [
        ("desktop", big_adapter()),
        ("downlevel", wgpu::Limits::downlevel_defaults()),
        // Below downlevel on several axes, including zero storage buffers.
        ("webgl2", wgpu::Limits::downlevel_webgl2_defaults()),
        (
            "tiny buffers",
            wgpu::Limits {
                max_buffer_size: 64 * MIB,
                max_storage_buffer_binding_size: 32 * MIB,
                ..wgpu::Limits::downlevel_defaults()
            },
        ),
    ] {
        let req = requested_device_limits(&adapter);
        let mut over = Vec::new();
        req.check_limits_with_fail_fn(&adapter, false, |name, want, have| {
            over.push(format!("{name}: requested {want}, adapter has {have}"));
        });
        assert!(over.is_empty(), "{label}: {over:?}");
    }
}

/// A real kernel over buffers larger than the old 256 MiB cap.
#[test]
fn a_subtract_over_buffers_larger_than_256_mib_matches_cpu() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let Ok(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    })) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    // 300 MiB per tensor.
    let len = (300 * MIB / 4) as usize;
    // Skip on the *adapter's* limits: a device capped below what the adapter
    // offers is the failure this test exists to catch.
    let limits = adapter.limits();
    if limits.max_buffer_size < (len * 4) as u64
        || limits.max_storage_buffer_binding_size < (len * 4) as u64
    {
        eprintln!("skipping: adapter limits are below a 300 MiB buffer");
        return;
    }
    let ctx = pollster::block_on(GpuContext::from_adapter(adapter)).expect("device");

    let shape = TensorShape::new(1, len / 1024, 1024);
    let a: CpuTensor<f32> =
        Tensor::from_vec((0..len).map(|i| (i % 4093) as f32).collect(), shape).unwrap();
    let b: CpuTensor<f32> =
        Tensor::from_vec((0..len).map(|i| (i % 127) as f32 * 0.5).collect(), shape).unwrap();
    let want = CpuBackend::new().unwrap().subtract(&a, &b).unwrap();

    let got = ctx
        .subtract(&a.to_gpu_ctx(&ctx).unwrap(), &b.to_gpu_ctx(&ctx).unwrap())
        .unwrap()
        .to_cpu_ctx(&ctx)
        .unwrap();

    let got = got.storage.as_slice().unwrap();
    let want = want.storage.as_slice().unwrap();
    assert_eq!(got.len(), want.len());
    // Exact: both sides compute one f32 subtraction of the same operands.
    let first_bad = got
        .iter()
        .zip(want)
        .position(|(g, w)| g.to_bits() != w.to_bits());
    assert_eq!(first_bad, None, "first mismatching element");
}
