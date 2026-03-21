struct Params {
    width: u32,
    height: u32,
}

@group(0) @binding(0) var<storage, read> vertex_map: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> normal_map: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

fn cross(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

fn normalize_safe(v: vec3<f32>) -> vec3<f32> {
    let len = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len > 0.0001) {
        return v / len;
    }
    return vec3<f32>(0.0, 1.0, 0.0);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let idx = y * params.width + x;
    let v = vertex_map[idx].xyz;

    // Check if vertex is valid
    if (vertex_map[idx].w < 0.5) {
        normal_map[idx] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return;
    }

    // Get neighbors (with boundary checks)
    let left_valid = x > 0u;
    let right_valid = x < params.width - 1u;
    let up_valid = y > 0u;
    let down_valid = y < params.height - 1u;

    var normal = vec3<f32>(0.0, 0.0, 0.0);
    var count = 0u;

    // Compute normals from neighboring triangles
    if (left_valid && down_valid) {
        let v_left = vertex_map[(y + 1u) * params.width + (x - 1u)].xyz;
        let v_down = vertex_map[(y + 1u) * params.width + x].xyz;
        let v_valid = vertex_map[(y + 1u) * params.width + (x - 1u)].w > 0.5 
                   && vertex_map[(y + 1u) * params.width + x].w > 0.5 
                   && vertex_map[idx].w > 0.5;
        if (v_valid) {
            let e1 = v_left - v;
            let e2 = v_down - v;
            normal = normal + cross(e1, e2);
            count = count + 1u;
        }
    }

    if (right_valid && down_valid) {
        let v_right = vertex_map[(y + 1u) * params.width + (x + 1u)].xyz;
        let v_down = vertex_map[(y + 1u) * params.width + x].xyz;
        let v_valid = vertex_map[(y + 1u) * params.width + (x + 1u)].w > 0.5 
                   && vertex_map[(y + 1u) * params.width + x].w > 0.5 
                   && vertex_map[idx].w > 0.5;
        if (v_valid) {
            let e1 = v_down - v;
            let e2 = v_right - v;
            normal = normal + cross(e1, e2);
            count = count + 1u;
        }
    }

    if (left_valid && up_valid) {
        let v_left = vertex_map[(y - 1u) * params.width + (x - 1u)].xyz;
        let v_up = vertex_map[(y - 1u) * params.width + x].xyz;
        let v_valid = vertex_map[(y - 1u) * params.width + (x - 1u)].w > 0.5 
                   && vertex_map[(y - 1u) * params.width + x].w > 0.5 
                   && vertex_map[idx].w > 0.5;
        if (v_valid) {
            let e1 = v_up - v;
            let e2 = v_left - v;
            normal = normal + cross(e1, e2);
            count = count + 1u;
        }
    }

    if (right_valid && up_valid) {
        let v_right = vertex_map[(y - 1u) * params.width + (x + 1u)].xyz;
        let v_up = vertex_map[(y - 1u) * params.width + x].xyz;
        let v_valid = vertex_map[(y - 1u) * params.width + (x + 1u)].w > 0.5 
                   && vertex_map[(y - 1u) * params.width + x].w > 0.5 
                   && vertex_map[idx].w > 0.5;
        if (v_valid) {
            let e1 = v_right - v;
            let e2 = v_up - v;
            normal = normal + cross(e1, e2);
            count = count + 1u;
        }
    }

    if (count > 0u) {
        normal_map[idx] = vec4<f32>(normalize_safe(normal), 0.0);
    } else {
        normal_map[idx] = vec4<f32>(0.0, 1.0, 0.0, 0.0);
    }
}
