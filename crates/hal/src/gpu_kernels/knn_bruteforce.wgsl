// Brute-force k-nearest-neighbours, on the GPU.
//
// One invocation per query point: it scans every point, keeping a top-k list
// sorted by ascending squared distance in private arrays. Brute force is
// O(num_points * k) per query - the point of this kernel is that the search
// runs *on the GPU* (the function used to read the whole cloud back to the
// host and loop there while claiming GPU); a Morton-traversal kernel is the
// future, this is the honest brute force.
//
// The list is capped at K_MAX entries: the host refuses k > 128 rather than
// silently doing something else.

struct Params {
    num_points: u32,
    k: u32,
    num_queries: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> queries: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const K_MAX: u32 = 128u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q = gid.x;
    if (q >= params.num_queries) {
        return;
    }
    let k = min(params.k, K_MAX);

    var best_d: array<f32, 128>;
    var best_i: array<u32, 128>;
    var count: u32 = 0u;

    let query = queries[q].xyz;
    for (var j = 0u; j < params.num_points; j = j + 1u) {
        let dv = points[j].xyz - query;
        let dist = dot(dv, dv);

        // Strict comparison at the full-k boundary keeps the *first* index on
        // exact distance ties, matching the CPU path's stable sort.
        if count == k && dist >= best_d[k - 1u] {
            continue;
        }

        var pos = count;
        if pos == k {
            pos = k - 1u;
        }
        while pos > 0u && best_d[pos - 1u] > dist {
            best_d[pos] = best_d[pos - 1u];
            best_i[pos] = best_i[pos - 1u];
            pos = pos - 1u;
        }
        best_d[pos] = dist;
        best_i[pos] = j;
        if count < k {
            count = count + 1u;
        }
    }

    // Out: interleaved (distance, index) pairs, ascending distance. Slots
    // beyond `min(k, num_points)` stay zero; the host trims them.
    let base = q * k * 2u;
    for (var i = 0u; i < count; i = i + 1u) {
        results[base + 2u * i] = best_d[i];
        results[base + 2u * i + 1u] = f32(best_i[i]);
    }
}
