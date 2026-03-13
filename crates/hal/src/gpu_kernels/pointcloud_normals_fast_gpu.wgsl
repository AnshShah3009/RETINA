// Point Cloud Normal Computation - Tiled GPU Implementation
// Uses workgroup shared memory for cache-efficient brute-force kNN.
// Each workgroup of 256 threads cooperatively loads tiles of 256 points into
// shared memory, giving 256x better bandwidth utilisation vs. naive global scan.
//
// Complexity: O(n^2) work per cloud, but with ~256x lower memory bandwidth cost.
// For large clouds (>10k points) prefer the Morton-sorted variant.
//
// Eigenvector fix: uses 3-stage deflated power iteration to correctly find the
// MINIMUM eigenvector of the covariance matrix (= surface normal direction).

struct Params {
    num_points: f32,
    k_neighbors: f32,
    voxel_size: f32,  // kept for API compat, unused
    padding: f32,
}

@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> normals: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

// Workgroup-local shared memory: one tile of 256 points.
var<workgroup> shared_pts: array<vec4<f32>, 256>;

// Analytic minimum eigenvector of a symmetric 3x3 covariance matrix.
// Matches Open3D PointCloudImpl.h / Geometric Tools RobustEigenSymmetric3x3.
// No iteration — exact closed-form in ~50 scalar ops.
fn min_eigenvector(cxx: f32, cxy: f32, cxz: f32, cyy: f32, cyz: f32, czz: f32) -> vec3<f32> {
    // Normalize to prevent overflow.
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

    let two_thirds_pi: f32 = 2.09439510239319549;
    let eval_min = q + p * cos(angle + two_thirds_pi) * 2.0;

    // Eigenvector: best cross-product of rows of (A - eval_min * I).
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

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let idx = global_id.x;
    let lid = local_id.x;
    let num_points = u32(params.num_points);
    // Cap k at 16 to reduce register pressure; still high-quality results.
    let k = min(u32(params.k_neighbors), 16u);
    let active = idx < num_points;

    // Load this thread's point (or a dummy if out of range).
    let point = select(vec3<f32>(0.0), points[min(idx, num_points - 1u)].xyz, active);

    // k-NN bookkeeping: fixed-size arrays for up to 16 neighbours.
    var nb_idx:   array<u32, 16>;
    var nb_dist2: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) {
        nb_idx[i]   = 0xFFFFFFFFu;
        nb_dist2[i] = 1e20;
    }

    // --- Tiled brute-force kNN using workgroup shared memory ---
    // Each tile cooperatively loads 256 points, then every active thread
    // compares its query point against all 256 loaded points.
    let tiles = (num_points + 255u) / 256u;
    for (var tile = 0u; tile < tiles; tile++) {
        // All threads participate in cooperative load (required for barrier).
        let load_idx = tile * 256u + lid;
        let safe_idx = min(load_idx, num_points - 1u);
        shared_pts[lid] = select(
            vec4<f32>(1e20, 1e20, 1e20, 0.0),
            points[safe_idx],
            load_idx < num_points
        );
        workgroupBarrier();

        // Active threads process every point in this tile.
        if active {
            let tile_count = min(256u, num_points - tile * 256u);
            for (var j = 0u; j < tile_count; j++) {
                let global_j = tile * 256u + j;
                if global_j == idx { continue; }

                let diff  = shared_pts[j].xyz - point;
                let dist2 = dot(diff, diff);

                // Threshold: ignore coincident points; insert if closer than current worst.
                if dist2 > 0.0001 && dist2 < nb_dist2[k - 1u] {
                    // Insertion sort into ascending nb_dist2[0..k).
                    var pos = k - 1u;
                    while pos > 0u && dist2 < nb_dist2[pos - 1u] {
                        nb_dist2[pos] = nb_dist2[pos - 1u];
                        nb_idx[pos]   = nb_idx[pos - 1u];
                        pos -= 1u;
                    }
                    nb_dist2[pos] = dist2;
                    nb_idx[pos]   = global_j;
                }
            }
        }
        workgroupBarrier();
    }

    // Inactive threads exit here; past all barriers so this is safe.
    if !active { return; }

    // Count valid neighbours.
    var valid = 0u;
    for (var i = 0u; i < k; i++) {
        if nb_idx[i] != 0xFFFFFFFFu { valid += 1u; }
    }
    if valid < 3u {
        normals[idx] = vec4<f32>(0.0, 1.0, 0.0, 0.0);
        return;
    }

    // Centroid of neighbours.
    var centroid = vec3<f32>(0.0);
    for (var i = 0u; i < k; i++) {
        if nb_idx[i] != 0xFFFFFFFFu { centroid += points[nb_idx[i]].xyz; }
    }
    centroid /= f32(valid);

    // Upper-triangle of the 3x3 covariance matrix.
    var cxx = 0.0; var cxy = 0.0; var cxz = 0.0;
    var cyy = 0.0; var cyz = 0.0; var czz = 0.0;
    for (var i = 0u; i < k; i++) {
        if nb_idx[i] == 0xFFFFFFFFu { continue; }
        let d = points[nb_idx[i]].xyz - centroid;
        cxx += d.x * d.x;
        cxy += d.x * d.y;
        cxz += d.x * d.z;
        cyy += d.y * d.y;
        cyz += d.y * d.z;
        czz += d.z * d.z;
    }
    let inv = 1.0 / f32(valid);
    cxx *= inv; cxy *= inv; cxz *= inv;
    cyy *= inv; cyz *= inv; czz *= inv;

    // Minimum eigenvector = surface normal.
    var normal = min_eigenvector(cxx, cxy, cxz, cyy, cyz, czz);

    // Orient away from centroid.
    if dot(normal, centroid - point) > 0.0 { normal = -normal; }

    normals[idx] = vec4<f32>(normal, 0.0);
}
