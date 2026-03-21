// GPU Point Cloud Morton Encoding Kernel

struct Params {
    min_bound: vec4<f32>,
    grid_size: f32,
    num_points: u32,
    padding1: u32,
    padding2: u32,
}

@group(0) @binding(0) var<storage, read>  points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> codes: array<u32>;
@group(0) @binding(2) var<storage, read_write> indices: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

fn morton_encode(x: u32, y: u32, z: u32) -> u32 {
    var mx = x & 0x000003FFu;
    var my = y & 0x000003FFu;
    var mz = z & 0x000003FFu;

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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_points) { return; }

    let p = points[idx].xyz;
    
    let x = u32(clamp((p.x - params.min_bound.x) / params.grid_size, 0.0, 1023.0));
    let y = u32(clamp((p.y - params.min_bound.y) / params.grid_size, 0.0, 1023.0));
    let z = u32(clamp((p.z - params.min_bound.z) / params.grid_size, 0.0, 1023.0));

    codes[idx] = morton_encode(x, y, z);
    indices[idx] = idx;
}
