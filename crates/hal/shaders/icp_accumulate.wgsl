// ICP Jacobian Accumulation Kernel
// Accumulates J^T * J (6x6) and J^T * r (6x1)

struct Params {
    num_points: u32,
    transform: mat4x4<f32>,
}

// Five storage buffers, not six.
//
// The portable default for `max_storage_buffers_per_shader_stage` is 8, but the
// device this was written for reports 4, and wgpu cannot derive an implicit
// layout past that: "Too many bindings of type StorageBuffers, limit is 4, count
// was 6". The pipeline therefore could never be created, so GPU ICP was a hard
// failure rather than a wrong answer. The two accumulators are merged into one
// buffer - `ata` occupies 36 slots and `atb` the remaining 6 - which brings the
// count to 5. `atb` is packed into the same atomics as `ata` rather than given
// its own binding.
// Four storage buffers, which is the device limit.
//
// Source and target are interleaved into one buffer as pairs - index `2*i` is
// the source point, `2*i + 1` the target - so the two `array<vec4<f32>>` become
// one. The accumulators are already merged for the same reason.
@group(0) @binding(0) var<storage, read> point_pairs: array<vec4<f32>>; // [2i] = source, [2i+1] = target
@group(0) @binding(1) var<storage, read> target_normals: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> correspondences: array<vec2<u32>>; // (src_idx, tgt_idx)
@group(0) @binding(3) var<storage, read_write> accum: array<atomic<u32>>; // [0..36) = JtJ, [36..42) = Jtr
@group(0) @binding(4) var<uniform> params: Params;

const ATA_SLOTS: u32 = 36u;
const ATB_SLOT0: u32 = 36u;

// Atomic float addition via compare-exchange on the bit-cast value.
// A previous revision accumulated i32(val * 1e6), which overflows the i32
// range at |sum| > ~2147 — J^T J entries summed over thousands of meter-scale
// correspondences blow past that and wrap to garbage.
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

    let p_src = point_pairs[src_idx * 2u].xyz;
    let p_tgt = point_pairs[tgt_idx * 2u + 1u].xyz;
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
{
            loop {
                let old = atomicLoad(&accum[ATB_SLOT0 + i]);
                let summed = bitcast<u32>(bitcast<f32>(old) + (J[i] * residual));
                let res = atomicCompareExchangeWeak(&accum[ATB_SLOT0 + i], old, summed);
                if (res.exchanged) { break; }
            }
        }
    }

    // Accumulate J^T * J (only upper triangle due to symmetry? no, full for simplicity first)
    for (var i = 0u; i < 6u; i++) {
        for (var j = 0u; j < 6u; j++) {
            {
                loop {
                    let idx = i * 6u + j;
                    let old = atomicLoad(&accum[idx]);
                    let summed = bitcast<u32>(bitcast<f32>(old) + (J[i] * J[j]));
                    let res = atomicCompareExchangeWeak(&accum[idx], old, summed);
                    if (res.exchanged) { break; }
                }
            }
        }
    }
}
