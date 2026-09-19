//! Regression tests for CPU/GPU-consistency and robustness bug fixes.
//!
//! Each test targets one specific fix so a regression is caught immediately.

use cv_core::storage::CpuStorage;
use cv_core::tensor::Tensor;
use cv_core::TensorShape;
use cv_hal::context::ComputeContext;
use cv_hal::cpu::CpuBackend;

fn get_cpu_backend() -> CpuBackend {
    CpuBackend::new().expect("CPU backend unavailable")
}

fn f32_tensor(data: Vec<f32>, c: usize, h: usize, w: usize) -> Tensor<f32, CpuStorage<f32>> {
    Tensor::from_vec(data, TensorShape::new(c, h, w)).unwrap()
}

// ---------------------------------------------------------------------------
// Finding 7: CPU Canny hysteresis must emit 255 (not 1) for strong edges.
// ---------------------------------------------------------------------------
#[test]
fn test_canny_strong_edges_are_255() {
    let cpu = get_cpu_backend();
    let (w, h) = (16usize, 16usize);
    let mut data = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 8..w {
            data[y * w + x] = 255.0;
        }
    }
    let input = f32_tensor(data, 1, h, w);

    let edges = cpu.canny(&input, 50.0f32, 100.0f32).unwrap();
    let out = edges.as_slice().unwrap();

    assert!(
        out.iter().any(|&v| v == 255.0),
        "expected at least one strong edge at 255"
    );
    assert!(
        out.iter().all(|&v| v == 0.0 || v == 255.0),
        "canny mask must be exactly 0/255"
    );
}

// ---------------------------------------------------------------------------
// Finding 6: CPU pyramid_down uses floor halving to match the GPU convention.
// ---------------------------------------------------------------------------
#[test]
fn test_pyramid_down_odd_dimensions_floor() {
    let cpu = get_cpu_backend();
    let input = f32_tensor(vec![1.0; 15], 1, 3, 5);
    let out = cpu.pyramid_down(&input).unwrap();
    assert_eq!(out.shape.width, 2);
    assert_eq!(out.shape.height, 1);
}

// ---------------------------------------------------------------------------
// Finding 4: optical flow must treat input points as level-0 coordinates and
// only double the estimate when moving to a finer level.
// ---------------------------------------------------------------------------
#[test]
fn test_optical_flow_multi_level_identity_keeps_points() {
    let cpu = get_cpu_backend();
    let (w, h) = (16usize, 16usize);
    let make = || {
        let data: Vec<f32> = (0..w * h).map(|i| ((i * 13) % 256) as f32).collect();
        f32_tensor(data, 1, h, w)
    };
    let f0 = make();
    let f1 = make();
    let l0 = cpu.pyramid_down(&f0).unwrap();
    let l1 = cpu.pyramid_down(&f1).unwrap();

    let pts = [[8.0f32, 8.0]];
    let res = cpu
        .optical_flow_lk(&[f0, l0], &[f1, l1], &pts, 5, 10)
        .unwrap();

    assert!(
        (res[0][0] - 8.0).abs() < 0.5,
        "multi-level identity moved x to {}",
        res[0][0]
    );
    assert!(
        (res[0][1] - 8.0).abs() < 0.5,
        "multi-level identity moved y to {}",
        res[0][1]
    );
}

// ---------------------------------------------------------------------------
// Finding 3: point-cloud normals must search in 3-D, not with a packed index
// that biases neighborhoods toward low indices.
// ---------------------------------------------------------------------------
#[test]
fn test_pointcloud_normals_uses_3d_neighborhood() {
    let cpu = get_cpu_backend();
    let n = 108usize;
    let mut pts = vec![0.0f32; n * 4];
    fn set(pts: &mut [f32], i: usize, p: [f32; 3]) {
        pts[i * 4] = p[0];
        pts[i * 4 + 1] = p[1];
        pts[i * 4 + 2] = p[2];
        pts[i * 4 + 3] = 1.0;
    }

    // Low-index group on the z = 0 plane, ~57 units from the query point.
    let group_a = [
        [40.0, 40.0, 0.0],
        [41.0, 40.0, 0.0],
        [39.0, 40.0, 0.0],
        [40.0, 41.0, 0.0],
        [40.0, 39.0, 0.0],
        [41.0, 41.0, 0.0],
        [39.0, 39.0, 0.0],
        [41.0, 39.0, 0.0],
    ];
    for (k, p) in group_a.iter().enumerate() {
        set(&mut pts, k, *p);
    }
    // Far-away filler points.
    for i in 8..100 {
        set(&mut pts, i, [1000.0 + i as f32, 1000.0, 1000.0]);
    }
    // High-index group on the x = 0 plane, clustered at the query.
    let group_b = [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 1.0],
        [0.0, -1.0, -1.0],
        [0.0, 1.0, -1.0],
    ];
    for (k, p) in group_b.iter().enumerate() {
        set(&mut pts, 100 + k, *p);
    }

    let pc = f32_tensor(pts, 4, n, 1);
    let normals = cpu.pointcloud_normals(&pc, 8).unwrap();
    let ns = normals.as_slice().unwrap();
    let (nx, ny, nz) = (ns[100 * 4], ns[100 * 4 + 1], ns[100 * 4 + 2]);
    assert!(
        nx.abs() > 0.9,
        "expected x-dominant normal for x=0 plane, got ({nx}, {ny}, {nz})"
    );
}

// ---------------------------------------------------------------------------
// Finding 1: marching cubes must not index an unset edge vertex. Cube index
// 0x21 (corners 0 and 5 inside) crosses edge 9, so the edge table needs 0x339.
// ---------------------------------------------------------------------------
#[test]
fn test_marching_cubes_no_unset_edge_vertex() {
    let cpu = get_cpu_backend();
    let (vx, vy, vz) = (2usize, 2usize, 2usize);
    let mut data = vec![0.0f32; vx * vy * vz * 2];
    for iz in 0..vz {
        for iy in 0..vy {
            for ix in 0..vx {
                let idx = iz * (vx * vy * 2) + (iy * vx + ix) * 2;
                let inside = (ix == 0 && iy == 0 && iz == 0) || (ix == 1 && iy == 0 && iz == 1);
                data[idx] = if inside { -1.0 } else { 1.0 };
                data[idx + 1] = 1.0;
            }
        }
    }
    let vol = f32_tensor(data, vz * 2, vy, vx);
    let verts = cpu.tsdf_extract_mesh(&vol, 1.0f32, 0.0f32, 1024).unwrap();

    assert!(
        !verts.is_empty(),
        "cube index 0x21 should produce a surface"
    );
    for v in &verts {
        let p = v.pos();
        assert!(
            !(p[0].abs() < 1e-6 && p[1].abs() < 1e-6 && p[2].abs() < 1e-6),
            "bogus vertex at origin from an unset edge vertex: {:?}",
            p
        );
    }
}

// ---------------------------------------------------------------------------
// Finding 8: subtract must validate shapes instead of panicking.
// ---------------------------------------------------------------------------
#[test]
fn test_subtract_shape_mismatch_returns_error() {
    let cpu = get_cpu_backend();
    let a = f32_tensor(vec![1.0, 2.0, 3.0, 4.0], 1, 2, 2);
    let b = f32_tensor(vec![1.0, 2.0], 1, 1, 2);

    assert!(
        cpu.subtract(&a, &b).is_err(),
        "mismatched shapes must return an error, not panic"
    );

    let b_ok = f32_tensor(vec![1.0, 1.0, 1.0, 1.0], 1, 2, 2);
    let ok = cpu.subtract(&a, &b_ok).unwrap();
    assert_eq!(ok.as_slice().unwrap(), &[0.0, 1.0, 2.0, 3.0]);
}

// ---------------------------------------------------------------------------
// Finding 9: spmv must validate CSR inputs instead of underflowing/panicking.
// ---------------------------------------------------------------------------
#[test]
fn test_spmv_empty_row_ptr_returns_error() {
    let cpu = get_cpu_backend();
    let x = f32_tensor(vec![1.0, 2.0], 1, 2, 1);
    let res = cpu.spmv::<f32, CpuStorage<f32>>(&[], &[], &[], &x);
    assert!(res.is_err(), "empty row_ptr must return an error");
}

#[test]
fn test_spmv_rejects_bad_col_index() {
    let cpu = get_cpu_backend();
    let x = f32_tensor(vec![1.0, 2.0], 1, 2, 1);
    // row_ptr = [0, 1, 2], but column index 5 is out of range (2 columns).
    let res = cpu.spmv::<f32, CpuStorage<f32>>(&[0, 1, 2], &[0, 5], &[1.0f32, 1.0], &x);
    assert!(
        res.is_err(),
        "out-of-range column index must return an error"
    );
}

#[test]
fn test_spmv_identity() {
    let cpu = get_cpu_backend();
    let x = f32_tensor(vec![3.0, 4.0], 1, 2, 1);
    let y = cpu
        .spmv::<f32, CpuStorage<f32>>(&[0, 1, 2], &[0, 1], &[1.0f32, 1.0], &x)
        .unwrap();
    assert_eq!(y.as_slice().unwrap(), &[3.0, 4.0]);
}
