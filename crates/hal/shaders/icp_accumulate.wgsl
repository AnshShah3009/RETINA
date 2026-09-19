// ICP Jacobian Accumulation Kernel
// Accumulates J^T * J (6x6) and J^T * r (6x1)

struct Params {
    num_points: u32,
    transform: mat4x4<f32>,
}

@group(0) @binding(0) var<storage, read> source_points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> target_points: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> target_normals: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> correspondences: array<vec2<u32>>; // (src_idx, tgt_idx)
@group(0) @binding(4) var<storage, read_write> ata_accum: array<atomic<u32>, 36>; // 6x6, bit-cast f32
@group(0) @binding(5) var<storage, read_write> atb_accum: array<atomic<u32>, 6>;  // 6x1, bit-cast f32
@group(0) @binding(6) var<uniform> params: Params;

// Atomic float addition via compare-exchange on the bit-cast value.
// A previous revision accumulated i32(val * 1e6), which overflows the i32
// range at |sum| > ~2147 — J^T J entries summed over thousands of meter-scale
// correspondences blow past that and wrap to garbage.
fn atomic_add_f32(accum: ptr<storage, array<atomic<u32>>, read_write>, addr: u32, val: f32) {
    loop {
        let old = atomicLoad(&(*accum)[addr]);
        let new = bitcast<u32>(bitcast<f32>(old) + val);
        let res = atomicCompareExchangeWeak(&(*accum)[addr], old, new);
        if (res.exchanged) {
            break;
        }
    }
}

fn atomicAddFloat(addr: u32, val: f32, is_ata: bool) {
    if (is_ata) {
        atomic_add_f32(&ata_accum, addr, val);
    } else {
        atomic_add_f32(&atb_accum, addr, val);
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let num_corr = arrayLength(&correspondences);
    if (idx >= num_corr) {
        return;
    }

    let corr = correspondences[idx];
    let src_idx = corr.x;
    let tgt_idx = corr.y;

    let p_src = source_points[src_idx].xyz;
    let p_tgt = target_points[tgt_idx].xyz;
    let n_tgt = target_normals[tgt_idx].xyz;

    // Transform source point
    let p_trans = (params.transform * vec4<f32>(p_src, 1.0)).xyz;
    
    let diff = p_trans - p_tgt;
    let residual = dot(diff, n_tgt);

    // Jacobian for point-to-plane: J = [n^T, (p x n)^T]
    let cross_prod = cross(p_trans, n_tgt);
    let J = array<f32, 6>(
        n_tgt.x, n_tgt.y, n_tgt.z,
        cross_prod.x, cross_prod.y, cross_prod.z
    );

    // Accumulate J^T * r
    for (var i = 0u; i < 6u; i++) {
        atomicAddFloat(i, J[i] * residual, false);
    }

    // Accumulate J^T * J (only upper triangle due to symmetry? no, full for simplicity first)
    for (var i = 0u; i < 6u; i++) {
        for (var j = 0u; j < 6u; j++) {
            atomicAddFloat(i * 6u + j, J[i] * J[j], true);
        }
    }
}
