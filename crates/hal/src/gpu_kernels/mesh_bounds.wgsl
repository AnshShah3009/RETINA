struct Params {
    num_vertices: u32,
}

@group(0) @binding(0) var<storage, read> vertices: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> bounds: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (params.num_vertices == 0u) {
        bounds[0] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        bounds[1] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return;
    }

    var min_bound = vertices[0].xyz;
    var max_bound = vertices[0].xyz;

    for (var i = 1u; i < params.num_vertices; i++) {
        let v = vertices[i].xyz;
        min_bound = min(min_bound, v);
        max_bound = max(max_bound, v);
    }

    bounds[0] = vec4<f32>(min_bound, 0.0);
    bounds[1] = vec4<f32>(max_bound, 0.0);
}
