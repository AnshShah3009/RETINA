// Point Cloud Normal Computation - GPU Optimized with Morton Code Spatial Index
// Uses Z-order curve (Morton code) for cache-efficient neighbor search

struct Params {
    num_points: f32,
    k_neighbors: f32,
    grid_size: f32,
    padding: f32,
}

@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> normals: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> sorted_indices: array<u32>;  // Morton-sorted indices
@group(0) @binding(3) var<storage, read> morton_codes: array<u32>;    // Morton codes
@group(0) @binding(4) var<uniform> params: Params;

// Convert 3D point to Morton code (Z-order curve)
fn morton3d(x: u32, y: u32, z: u32) -> u32 {
    var mx = x;
    var my = y;
    var mz = z;
    
    // Expand bits: 00000000xxxxxxxx -> 0000x0x0x0x0x0x0x
    mx = (mx | (mx << 16u)) & 0x030000FFu;
    mx = (mx | (mx << 8u)) & 0x0300F00Fu;
    mx = (mx | (mx << 4u)) & 0x030C30C3u;
    mx = (mx | (mx << 2u)) & 0x09249249u;
    
    my = (my | (my << 16u)) & 0x030000FFu;
    my = (my | (my << 8u)) & 0x0300F00Fu;
    my = (my | (my << 4u)) & 0x030C30C3u;
    my = (my | (my << 2u)) & 0x09249249u;
    
    mz = (mz | (mz << 16u)) & 0x030000FFu;
    mz = (mz | (mz << 8u)) & 0x0300F00Fu;
    mz = (mz | (mz << 4u)) & 0x030C30C3u;
    mz = (mz | (mz << 2u)) & 0x09249249u;
    
    return mx | (my << 1u) | (mz << 2u);
}

// Binary search for Morton code range
fn find_morton_range(code: u32, num: u32) -> vec2<u32> {
    // Find left bound
    var lo = 0u;
    var hi = num;
    while lo < hi {
        let mid = (lo + hi) / 2u;
        if morton_codes[mid] < code {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    let left = lo;
    
    // Find right bound (search for code + 1)
    hi = num;
    while lo < hi {
        let mid = (lo + hi) / 2u;
        if morton_codes[mid] <= code {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    
    return vec2<u32>(left, lo);
}

// Find k nearest neighbors using Morton-sorted data
fn find_neighbors_sorted(
    point: vec3<f32>,
    idx_in_sorted: u32,
    num_points: u32,
    k: u32,
) -> array<u32, 32> {
    var neighbors: array<u32, 32>;
    var dists: array<f32, 32>;
    
    // Initialize
    for (var i = 0u; i < k; i = i + 1u) {
        dists[i] = 1e20;
        neighbors[i] = 0xFFFFFFFFu;
    }
    
    // Search in expanding windows around sorted position
    // This exploits spatial locality from Morton ordering
    let search_radius = min(1024u, num_points);
    let start = max(0u, idx_in_sorted - search_radius);
    let end = min(num_points, idx_in_sorted + search_radius);
    
    for (var j = start; j < end; j = j + 1u) {
        if j == idx_in_sorted { continue; }
        
        let other_idx = sorted_indices[j];
        let other = points[other_idx].xyz;
        let diff = point - other;
        let d = dot(diff, diff);
        
        // Insert into sorted list
        if d < dists[k - 1u] {
            var insert_pos = k;
            while insert_pos > 0u && d < dists[insert_pos - 1u] {
                insert_pos = insert_pos - 1u;
            }
            if insert_pos < k {
                // Shift
                for (var m = k - 1u; m > insert_pos; m = m - 1u) {
                    dists[m] = dists[m - 1u];
                    neighbors[m] = neighbors[m - 1u];
                }
                dists[insert_pos] = d;
                neighbors[insert_pos] = other_idx;
            }
        }
    }
    
    return neighbors;
}

// Analytic minimum eigenvector of a symmetric 3x3 covariance matrix.
// Matches Open3D PointCloudImpl.h / Geometric Tools RobustEigenSymmetric3x3.
// No iteration needed — exact closed-form solution in ~50 scalar ops.
//
// Algorithm:
//   1. Normalize by max coefficient (numerical stability).
//   2. Compute eigenvalues via trigonometric method (Cardano/Smith).
//   3. For minimum eigenvalue: eigenvector = best cross-product of
//      the rows of (A - lambda_min * I).
fn analytic_min_eigenvector(
    cxx: f32, cxy: f32, cxz: f32,
    cyy: f32, cyz: f32, czz: f32,
) -> vec3<f32> {
    // Normalize to avoid overflow / underflow.
    let max_c = max(max(abs(cxx), max(abs(cxy), abs(cxz))),
                   max(abs(cyy), max(abs(cyz), abs(czz))));
    if max_c < 1e-30 { return vec3<f32>(0.0, 0.0, 1.0); }
    let s = 1.0 / max_c;
    let a00 = cxx * s;  let a01 = cxy * s;  let a02 = cxz * s;
    let a11 = cyy * s;  let a12 = cyz * s;  let a22 = czz * s;

    // Off-diagonal norm squared.
    let norm = a01 * a01 + a02 * a02 + a12 * a12;

    // Shift: q = trace / 3.
    let q = (a00 + a11 + a22) / 3.0;
    let b00 = a00 - q;  let b11 = a11 - q;  let b22 = a22 - q;

    // Scale of deviatoric part.
    let p = sqrt((b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * norm) / 6.0);
    if p < 1e-10 { return vec3<f32>(0.0, 0.0, 1.0); }

    // Determinant of (A - q*I) / p — should lie in [-1, 1].
    let c00 = b11 * b22 - a12 * a12;
    let c01 = a01 * b22 - a12 * a02;
    let c02 = a01 * a12 - b11 * a02;
    let det = (b00 * c00 - a01 * c01 + a02 * c02) / (p * p * p);

    let half_det = clamp(det * 0.5, -1.0, 1.0);
    let angle = acos(half_det) / 3.0;

    // Minimum eigenvalue corresponds to angle + 2π/3.
    let two_thirds_pi: f32 = 2.09439510239319549;
    let eval_min = q + p * cos(angle + two_thirds_pi) * 2.0;

    // Eigenvector for eval_min: use rows of (A - eval_min * I),
    // pick the cross-product pair with the largest squared magnitude.
    let r0 = vec3<f32>(a00 - eval_min, a01, a02);
    let r1 = vec3<f32>(a01, a11 - eval_min, a12);
    let r2 = vec3<f32>(a02, a12, a22 - eval_min);

    let r0xr1 = cross(r0, r1);
    let r0xr2 = cross(r0, r2);
    let r1xr2 = cross(r1, r2);

    let d0 = dot(r0xr1, r0xr1);
    let d1 = dot(r0xr2, r0xr2);
    let d2 = dot(r1xr2, r1xr2);

    var best: vec3<f32>;
    if d0 >= d1 && d0 >= d2      { best = r0xr1; }
    else if d1 >= d2             { best = r0xr2; }
    else                         { best = r1xr2; }

    let blen = length(best);
    if blen < 1e-10 { return vec3<f32>(0.0, 0.0, 1.0); }
    return best / blen;
}

// Compute normal using PCA with power iteration
fn compute_normal_pca(point: vec3<f32>, neighbors: array<u32, 32>, k: u32) -> vec3<f32> {
    // Count valid
    var count = 0u;
    for (var i = 0u; i < k; i = i + 1u) {
        if neighbors[i] != 0xFFFFFFFFu { count = count + 1u; }
    }
    
    if count < 3u { return vec3<f32>(0.0, 0.0, 1.0); }
    
    // Centroid
    var centroid = vec3<f32>(0.0);
    for (var i = 0u; i < k; i = i + 1u) {
        if neighbors[i] != 0xFFFFFFFFu {
            centroid = centroid + points[neighbors[i]].xyz;
        }
    }
    centroid = centroid / f32(count);
    
    // Covariance (upper triangle only)
    var cov_xx = 0.0f; var cov_xy = 0.0f; var cov_xz = 0.0f;
    var cov_yy = 0.0f; var cov_yz = 0.0f; var cov_zz = 0.0f;
    
    for (var i = 0u; i < k; i = i + 1u) {
        if neighbors[i] == 0xFFFFFFFFu { continue; }
        let d = points[neighbors[i]].xyz - centroid;
        cov_xx = cov_xx + d.x * d.x;
        cov_xy = cov_xy + d.x * d.y;
        cov_xz = cov_xz + d.x * d.z;
        cov_yy = cov_yy + d.y * d.y;
        cov_yz = cov_yz + d.y * d.z;
        cov_zz = cov_zz + d.z * d.z;
    }
    
    let inv_n = 1.0 / f32(count);
    cov_xx = cov_xx * inv_n; cov_xy = cov_xy * inv_n; cov_xz = cov_xz * inv_n;
    cov_yy = cov_yy * inv_n; cov_yz = cov_yz * inv_n; cov_zz = cov_zz * inv_n;
    
    // Analytic minimum eigenvector (Open3D / Geometric Tools algorithm).
    var normal = analytic_min_eigenvector(cov_xx, cov_xy, cov_xz, cov_yy, cov_yz, cov_zz);

    // Orient away from centroid.
    if dot(normal, centroid - point) > 0.0 {
        normal = -normal;
    }

    return normal;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let num_points = u32(params.num_points);
    let k = min(u32(params.k_neighbors), 32u);
    
    if idx >= num_points { return; }
    
    // Get sorted position for this point
    let sorted_idx = idx;  // We process in sorted order
    
    let point_idx = sorted_indices[idx];
    let point = points[point_idx].xyz;
    
    // Find neighbors using Morton-sorted spatial index
    let neighbors = find_neighbors_sorted(point, sorted_idx, num_points, k);
    
    // Compute normal
    let normal = compute_normal_pca(point, neighbors, k);
    
    normals[point_idx] = vec4<f32>(normal, 0.0);
}
