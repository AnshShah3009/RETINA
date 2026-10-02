use crate::context::{BorderMode, Interpolation};
use crate::gpu::GpuContext;
use crate::storage::GpuStorage;
use crate::Result;
use cv_core::{CameraIntrinsics, Distortion, Tensor, TensorShape};
use nalgebra::Matrix3;
use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct UndistortCameraParams {
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    ifx: f32,
    ify: f32,
    k1: f32,
    k2: f32,
    p1: f32,
    p2: f32,
    k3: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct UndistortImageParams {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    interpolation: u32,
    border_mode: u32,
    border_val: f32,
    _pad: u32,
}

/// The matrix the undistort shader multiplies a destination pixel by:
/// `Inv(R) · Inv(NewK)`.
///
/// **Both inverses are checked, and a singular input is an error rather than an
/// identity.**
///
/// The identity is not a neutral fallback here - it is a *valid* matrix meaning
/// "this camera has no intrinsics", so every destination pixel is used as its own
/// normalized coordinate and the rectification is silently skipped. The call
/// returns `Ok`, so the caller has a wrongly-warped image and no way to tell.
///
/// This is the GPU twin of the defect fixed in `cv-calib3d`'s
/// `init_undistort_rectify_map`, where the same `unwrap_or(Matrix3::identity())`
/// collapsed every destination pixel onto the principal point while reporting
/// 100% valid. `CameraIntrinsics` has public fields and a constructor that
/// validates nothing, so `fx = 0` - which makes `NewK` singular, since
/// `det = fx · fy` - is ordinary API use rather than an exotic input.
/// `rectification` is a caller-supplied matrix with no shape or orthonormality
/// check either.
///
/// Split out from [`undistort`] so this is testable without a GPU adapter: the
/// matrix logic is the whole of the defect, and a CI runner has no adapter.
fn rectification_matrix(
    new_intrinsics: &CameraIntrinsics,
    rectification: &Matrix3<f64>,
) -> Result<Matrix3<f64>> {
    let inv_new_k = new_intrinsics.try_inverse_matrix().ok_or_else(|| {
        crate::Error::InvalidInput(format!(
            "new_intrinsics is singular (fx = {}, fy = {}), so the intrinsic matrix \
             has no inverse and the undistort map is undefined; the identity would \
             silently drop the intrinsics instead of reporting this",
            new_intrinsics.fx, new_intrinsics.fy
        ))
    })?;
    let inv_r = rectification.try_inverse().ok_or_else(|| {
        crate::Error::InvalidInput(
            "rectification matrix is singular, so the rectified frame is undefined".to_string(),
        )
    })?;
    Ok(inv_r * inv_new_k)
}

pub fn undistort(
    ctx: &GpuContext,
    input: &Tensor<f32, GpuStorage<f32>>,
    intrinsics: &CameraIntrinsics,
    distortion: &Distortion,
    rectification: &Matrix3<f64>,
    new_intrinsics: &CameraIntrinsics,
    interpolation: Interpolation,
    border_mode: BorderMode<f32>,
) -> Result<Tensor<f32, GpuStorage<f32>>> {
    let (src_h, src_w) = input.shape.hw();
    let dst_w = new_intrinsics.width;
    let dst_h = new_intrinsics.height;
    let c = input.shape.channels;

    if c != 1 {
        return Err(crate::Error::NotSupported(
            "GPU Undistort currently only for grayscale".into(),
        ));
    }

    let out_len = (dst_w * dst_h) as usize;
    let byte_size = (out_len * 4) as u64;
    let usages =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let output_buffer = ctx.get_buffer(byte_size, usages);

    let cam_params = UndistortCameraParams {
        fx: intrinsics.fx as f32,
        fy: intrinsics.fy as f32,
        cx: intrinsics.cx as f32,
        cy: intrinsics.cy as f32,
        ifx: 1.0 / (intrinsics.fx as f32),
        ify: 1.0 / (intrinsics.fy as f32),
        k1: distortion.k1 as f32,
        k2: distortion.k2 as f32,
        p1: distortion.p1 as f32,
        p2: distortion.p2 as f32,
        k3: distortion.k3 as f32,
        _pad: 0,
    };

    let (b_mode, b_val) = match border_mode {
        BorderMode::Constant(v) => (0, v),
        BorderMode::Replicate => (1, 0.0),
        BorderMode::Wrap => (2, 0.0),
        BorderMode::Reflect => (3, 0.0),
        BorderMode::Reflect101 => (4, 0.0),
    };

    let img_params = UndistortImageParams {
        src_w: src_w as u32,
        src_h: src_h as u32,
        dst_w,
        dst_h,
        interpolation: match interpolation {
            Interpolation::Nearest => 0,
            _ => 1, // Default to bilinear for now
        },
        border_mode: b_mode,
        border_val: b_val,
        _pad: 0,
    };

    // rect_mat = Inv(R) * Inv(NewK); the shader then does norm_pt = rect_mat * dst_pt.
    let rect_mat_f64 = rectification_matrix(new_intrinsics, rectification)?;

    let mut rect_mat_data = [0.0f32; 12];
    for c in 0..3 {
        for r in 0..3 {
            rect_mat_data[c * 4 + r] = rect_mat_f64[(r, c)] as f32;
        }
    }

    let cam_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Undistort Cam Params"),
            contents: bytemuck::bytes_of(&cam_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let img_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Undistort Image Params"),
            contents: bytemuck::bytes_of(&img_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let rect_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Undistort Rect Mat"),
            contents: bytemuck::cast_slice(&rect_mat_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let shader_source = include_str!("../../shaders/undistort.wgsl");
    let pipeline = ctx.create_compute_pipeline(shader_source, "main");

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Undistort Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.storage.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: cam_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: img_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: rect_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let x = dst_w.div_ceil(4).div_ceil(16);
        let y = dst_h.div_ceil(16);
        pass.dispatch_workgroups(x, y, 1);
    }
    ctx.submit(encoder);

    Ok(Tensor {
        storage: GpuStorage::from_buffer(Arc::new(output_buffer), out_len),
        shape: TensorShape::new(c, dst_h as usize, dst_w as usize),
        dtype: input.dtype,
        _phantom: PhantomData,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k(fx: f64, fy: f64, cx: f64, cy: f64) -> CameraIntrinsics {
        CameraIntrinsics::new(fx, fy, cx, cy, 640, 480)
    }

    /// A zero focal length must be reported, not silently replaced by an identity.
    ///
    /// Established against the unfixed expression, evaluated directly because the
    /// old code was inline in `undistort` and cannot be reached without a GPU:
    ///
    /// ```text
    /// CameraIntrinsics::new(0.0, 400.0, 320.0, 240.0, ..).matrix().try_inverse()
    ///     -> None
    /// .unwrap_or(Matrix3::identity())
    ///     -> [[1,0,0],[0,1,0],[0,0,1]]
    /// ```
    ///
    /// So `rect_mat = inv_r * I = inv_r`, the intrinsics normalisation vanished
    /// entirely, and the call returned `Ok` with a wrongly-warped image.
    #[test]
    fn a_zero_focal_length_is_an_error_not_an_identity() {
        let err = rectification_matrix(&k(0.0, 400.0, 320.0, 240.0), &Matrix3::identity())
            .expect_err("fx = 0 makes NewK singular, so there is no undistort map");
        let msg = format!("{err}");
        assert!(
            msg.contains("singular") && msg.contains("fx"),
            "the error must name the problem and the offending parameter; got: {msg}"
        );

        // fy is the other factor of det = fx * fy, so it must be caught too.
        let err = rectification_matrix(&k(500.0, 0.0, 320.0, 240.0), &Matrix3::identity())
            .expect_err("fy = 0 makes NewK singular as well");
        assert!(format!("{err}").contains("fy"), "got: {err}");
    }

    /// A singular rectification matrix is equally unrecoverable, and was equally
    /// silently replaced by an identity.
    #[test]
    fn a_singular_rectification_matrix_is_an_error() {
        let err = rectification_matrix(&k(500.0, 400.0, 320.0, 240.0), &Matrix3::zeros())
            .expect_err("the zero matrix has no inverse, so the rectified frame is undefined");
        assert!(
            format!("{err}").contains("rectification"),
            "the error must name which matrix was singular; got: {err}"
        );
    }

    /// CONTROL: valid input still produces the analytic result.
    ///
    /// `NewK = [[fx,0,cx],[0,fy,cy],[0,0,1]]`, so its inverse is
    /// `[[1/fx, 0, -cx/fx], [0, 1/fy, -cy/fy], [0, 0, 1]]`. With an identity
    /// rectification the returned matrix is exactly that.
    #[test]
    fn valid_inputs_still_produce_the_analytic_inverse() {
        let new_k = k(500.0, 400.0, 320.0, 240.0);
        let m = rectification_matrix(&new_k, &Matrix3::identity())
            .expect("well-conditioned intrinsics must succeed");

        let expect = Matrix3::new(
            1.0 / 500.0,
            0.0,
            -320.0 / 500.0,
            0.0,
            1.0 / 400.0,
            -240.0 / 400.0,
            0.0,
            0.0,
            1.0,
        );
        assert!(
            (m - expect).norm() < 1e-12,
            "inverse of NewK is wrong:\n  got    {m:?}\n  expect {expect:?}"
        );

        // The property that matters: the destination principal point maps to the
        // origin, and one focal length to the right maps to exactly 1.
        let origin = m * nalgebra::Vector3::new(320.0, 240.0, 1.0);
        assert!(
            (origin[0]).abs() < 1e-12 && (origin[1]).abs() < 1e-12,
            "the principal point must map to the origin; got {origin:?}"
        );
        let right = m * nalgebra::Vector3::new(820.0, 240.0, 1.0);
        assert!(
            (right[0] - 1.0).abs() < 1e-12,
            "one focal length right of the principal point must map to x = 1; got {right:?}"
        );
        let down = m * nalgebra::Vector3::new(320.0, 640.0, 1.0);
        assert!(
            (down[1] - 1.0).abs() < 1e-12,
            "one focal length below must map to y = 1; got {down:?}"
        );
    }

    /// A non-trivial rectification must be applied, not ignored: the returned
    /// matrix is `Inv(R) * Inv(NewK)`, so a scaled R must scale the result.
    #[test]
    fn the_rectification_is_actually_composed_in() {
        let new_k = k(500.0, 400.0, 320.0, 240.0);
        let identity =
            rectification_matrix(&new_k, &Matrix3::identity()).expect("identity rectification");
        let scaled = rectification_matrix(&new_k, &(Matrix3::identity() * 2.0))
            .expect("a scaled rotation is still invertible");

        assert!(
            (scaled - identity * 0.5).norm() < 1e-12,
            "Inv(2I) = 0.5I, so the result must be halved; the rectification is \
             being dropped if these match:\n  identity {identity:?}\n  scaled   {scaled:?}"
        );
        assert!(
            (scaled - identity).norm() > 1e-6,
            "the test above is vacuous if the two are equal"
        );
    }
}
