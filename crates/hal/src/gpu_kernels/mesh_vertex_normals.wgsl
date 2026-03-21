struct Params {
    num_vertices: u32,
    num_faces: u32,
}

@group(0) @binding(0) var<storage, read> vertices: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> faces: array<vec3<u32>>;
@group(0) @binding(2) var<storage, read_write> normals: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: Params;

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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let vertex_idx = global_id.x;
    if (vertex_idx >= params.num_vertices) {
        return;
    }

    var normal = vec3<f32>(0.0, 0.0, 0.0);
    var count = 0u;

    // Accumulate face normals for this vertex
    for (var i = 0u; i < params.num_faces; i++) {
        let face = faces[i];
        
        // Check if this vertex is part of the face
        var is_member = false;
        if (face.x == vertex_idx || face.y == vertex_idx || face.z == vertex_idx) {
            is_member = true;
        }

        if (is_member) {
            let v0 = vertices[face.x].xyz;
            let v1 = vertices[face.y].xyz;
            let v2 = vertices[face.z].xyz;

            // Compute face normal using cross product
            let edge1 = v1 - v0;
            let edge2 = v2 - v0;
            let face_normal = cross(edge1, edge2);

            normal = normal + face_normal;
            count = count + 1u;
        }
    }

    // Normalize the accumulated normal
    let final_normal = normalize_safe(normal);
    
    normals[vertex_idx] = vec4<f32>(final_normal.x, final_normal.y, final_normal.z, 0.0);
}
