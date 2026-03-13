// Point Cloud Normals — Batch PCA (Hybrid CPU-kNN + GPU Eigenvector)
//
// This shader is the GPU half of the hybrid normal estimation pipeline:
//   CPU:  parallel voxel-hash kNN → per-point covariance matrices   (O(nk))
//   GPU:  batch analytic eigenvectors for all n covariances at once  (O(n))
//
// Layout: each covariance is packed as two vec4s (8 f32, 6 meaningful):
//   covs[2i + 0] = (cxx, cxy, cxz, cyy)
//   covs[2i + 1] = (cyz, czz, 0,   0  )
//
// All n eigenvector computations are fully independent → perfect GPU utilisation.

@group(0) @binding(0) var<storage, read>       covs:       array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> normals:    array<vec4<f32>>;
@group(0) @binding(2) var<uniform>             num_points: u32;

// Analytic minimum eigenvector of a symmetric 3x3 covariance matrix.
// Algorithm: Open3D PointCloudImpl.h / Geometric Tools RobustEigenSymmetric3x3.
// Eigenvalues via trigonometric (Cardano) method; eigenvector via best cross product.
fn analytic_min_eigenvector(
    cxx: f32, cxy: f32, cxz: f32,
    cyy: f32, cyz: f32, czz: f32,
) -> vec3<f32> {
    let max_c = max(max(abs(cxx), max(abs(cxy), abs(cxz))),
                   max(abs(cyy), max(abs(cyz), abs(czz))));
    if max_c < 1e-30 { return vec3<f32>(0.0, 0.0, 1.0); }
    let s = 1.0 / max_c;
    let a00 = cxx * s;  let a01 = cxy * s;  let a02 = cxz * s;
    let a11 = cyy * s;  let a12 = cyz * s;  let a22 = czz * s;

    let norm = a01 * a01 + a02 * a02 + a12 * a12;
    let q    = (a00 + a11 + a22) / 3.0;
    let b00  = a00 - q;  let b11 = a11 - q;  let b22 = a22 - q;
    let p    = sqrt((b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * norm) / 6.0);
    if p < 1e-10 { return vec3<f32>(0.0, 0.0, 1.0); }

    let c00      = b11 * b22 - a12 * a12;
    let c01      = a01 * b22 - a12 * a02;
    let c02      = a01 * a12 - b11 * a02;
    let det      = (b00 * c00 - a01 * c01 + a02 * c02) / (p * p * p);
    let half_det = clamp(det * 0.5, -1.0, 1.0);
    let angle    = acos(half_det) / 3.0;

    // Minimum eigenvalue: q + p * cos(angle + 2π/3) * 2.
    let two_thirds_pi: f32 = 2.09439510239319549;
    let eval_min = q + p * cos(angle + two_thirds_pi) * 2.0;

    // Eigenvector: best cross-product of rows of (A - eval_min * I).
    let r0 = vec3<f32>(a00 - eval_min, a01,            a02);
    let r1 = vec3<f32>(a01,            a11 - eval_min, a12);
    let r2 = vec3<f32>(a02,            a12,            a22 - eval_min);

    let r0xr1 = cross(r0, r1);
    let r0xr2 = cross(r0, r2);
    let r1xr2 = cross(r1, r2);

    let d0 = dot(r0xr1, r0xr1);
    let d1 = dot(r0xr2, r0xr2);
    let d2 = dot(r1xr2, r1xr2);

    var best: vec3<f32>;
    if d0 >= d1 && d0 >= d2 { best = r0xr1; }
    else if d1 >= d2        { best = r0xr2; }
    else                    { best = r1xr2; }

    let blen = length(best);
    if blen < 1e-10 { return vec3<f32>(0.0, 0.0, 1.0); }
    return best / blen;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= num_points { return; }

    // Unpack two vec4s → six covariance coefficients.
    let v0 = covs[idx * 2u + 0u];   // (cxx, cxy, cxz, cyy)
    let v1 = covs[idx * 2u + 1u];   // (cyz, czz,  _,   _ )

    let normal = analytic_min_eigenvector(v0.x, v0.y, v0.z, v0.w, v1.x, v1.y);
    normals[idx] = vec4<f32>(normal, 0.0);
}
