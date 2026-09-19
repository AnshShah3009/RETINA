// GPU Point Cloud Bounding Box Kernel

struct Bounds {
    min_pt: vec4<f32>,
    max_pt: vec4<f32>,
}

@group(0) @binding(0) var<storage, read>  points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> partial_bounds: array<Bounds>;
@group(0) @binding(2) var<uniform> num_points: u32;

var<workgroup> shared_min: array<vec3<f32>, 256>;
var<workgroup> shared_max: array<vec3<f32>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) group_id: vec3<u32>) {
    let gid = global_id.x;
    let tid = local_id.x;
    
    var local_min = vec3<f32>(1e20);
    var local_max = vec3<f32>(-1e20);

    if (gid < num_points) {
        let p = points[gid].xyz;
        local_min = p;
        local_max = p;
    }

    shared_min[tid] = local_min;
    shared_max[tid] = local_max;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_min[tid] = min(shared_min[tid], shared_min[tid + s]);
            shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        partial_bounds[group_id.x].min_pt = vec4<f32>(shared_min[0], 0.0);
        partial_bounds[group_id.x].max_pt = vec4<f32>(shared_max[0], 0.0);
    }
}
