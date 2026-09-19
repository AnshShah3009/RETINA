// kdtree_build.wgsl
// GPU Morton code calculation for point cloud

@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> morton_codes: array<u32>;
@group(0) @binding(2) var<storage, read_write> indices: array<u32>;

struct Bounds {
    min_pt: vec4<f32>,
    max_pt: vec4<f32>,
};
@group(0) @binding(3) var<uniform> bounds: Bounds;

fn expand_bits(v: u32) -> u32 {
    var x = v & 0x000003FFu;
    x = (x | (x << 16)) & 0x030000FFu;
    x = (x | (x << 8)) & 0x0300F00Fu;
    x = (x | (x << 4)) & 0x030C30C3u;
    x = (x | (x << 2)) & 0x09249249u;
    return x;
}

fn calculate_morton(p: vec3<f32>) -> u32 {
    let size = bounds.max_pt.xyz - bounds.min_pt.xyz;
    let normalized = (p - bounds.min_pt.xyz) / size;
    let grid_coords = vec3<u32>(clamp(normalized * 1023.0, vec3<f32>(0.0), vec3<f32>(1023.0)));
    
    return (expand_bits(grid_coords.x) << 2u) | (expand_bits(grid_coords.y) << 1u) | expand_bits(grid_coords.z);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&points)) {
        return;
    }
    
    morton_codes[idx] = calculate_morton(points[idx].xyz);
    indices[idx] = idx;
}
