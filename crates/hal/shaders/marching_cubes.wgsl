// GPU Marching Cubes
// Extracts an isosurface mesh from the TSDF volume.

struct Params {
    vol_x: u32,
    vol_y: u32,
    vol_z: u32,
    voxel_size: f32,
    iso_level: f32,
    max_triangles: u32,
}

struct Voxel {
    tsdf: f32,
    weight: f32,
}

struct Vertex {
    pos: vec4<f32>, // w is padding
    norm: vec4<f32>, // w is padding
}

// Bindings
@group(0) @binding(0) var<storage, read> voxels: array<Voxel>;
@group(0) @binding(1) var<storage, read_write> vertices: array<Vertex>; 
@group(0) @binding(2) var<storage, read_write> counter: atomic<u32>;    
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> tables: array<i32>; // Combined edge and tri tables

// Vertex interpolation
fn vertex_interp(p1: vec3<f32>, val1: f32, p2: vec3<f32>, val2: f32) -> vec3<f32> {
    if (abs(params.iso_level - val1) < 0.00001) { return p1; }
    if (abs(params.iso_level - val2) < 0.00001) { return p2; }
    if (abs(val1 - val2) < 0.00001) { return p1; }
    
    let mu = (params.iso_level - val1) / (val2 - val1);
    return p1 + mu * (p2 - p1);
}

fn get_val(x: u32, y: u32, z: u32) -> f32 {
    let idx = z * params.vol_x * params.vol_y + y * params.vol_x + x;
    return voxels[idx].tsdf;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;

    if (x >= params.vol_x - 1u || y >= params.vol_y - 1u || z >= params.vol_z - 1u) {
        return;
    }

    let p0 = vec3<u32>(x, y, z);
    let p1 = vec3<u32>(x+1u, y, z);
    let p2 = vec3<u32>(x+1u, y+1u, z); 
    let p3 = vec3<u32>(x, y+1u, z);    
    let p4 = vec3<u32>(x, y, z+1u);
    let p5 = vec3<u32>(x+1u, y, z+1u);
    let p6 = vec3<u32>(x+1u, y+1u, z+1u);
    let p7 = vec3<u32>(x, y+1u, z+1u);

    let v0 = get_val(p0.x, p0.y, p0.z);
    let v1 = get_val(p1.x, p1.y, p1.z);
    let v2 = get_val(p2.x, p2.y, p2.z);
    let v3 = get_val(p3.x, p3.y, p3.z);
    let v4 = get_val(p4.x, p4.y, p4.z);
    let v5 = get_val(p5.x, p5.y, p5.z);
    let v6 = get_val(p6.x, p6.y, p6.z);
    let v7 = get_val(p7.x, p7.y, p7.z);

    var cube_index = 0u;
    if (v0 < params.iso_level) { cube_index |= 1u; }
    if (v1 < params.iso_level) { cube_index |= 2u; }
    if (v2 < params.iso_level) { cube_index |= 4u; }
    if (v3 < params.iso_level) { cube_index |= 8u; }
    if (v4 < params.iso_level) { cube_index |= 16u; }
    if (v5 < params.iso_level) { cube_index |= 32u; }
    if (v6 < params.iso_level) { cube_index |= 64u; }
    if (v7 < params.iso_level) { cube_index |= 128u; }

    let edge_table_ptr = 0u; // Start of edge table
    let tri_table_ptr = 256u; // Start of tri table
    
    let edges = u32(tables[edge_table_ptr + cube_index]);
    if (edges == 0u) { return; }

    var vertlist: array<vec3<f32>, 12>;
    let pos = vec3<f32>(f32(x) * params.voxel_size, f32(y) * params.voxel_size, f32(z) * params.voxel_size);
    let vs = params.voxel_size;

    let vp0 = pos + vec3<f32>(0.0, 0.0, 0.0);
    let vp1 = pos + vec3<f32>(vs, 0.0, 0.0);
    let vp2 = pos + vec3<f32>(vs, vs, 0.0);
    let vp3 = pos + vec3<f32>(0.0, vs, 0.0);
    let vp4 = pos + vec3<f32>(0.0, 0.0, vs);
    let vp5 = pos + vec3<f32>(vs, 0.0, vs);
    let vp6 = pos + vec3<f32>(vs, vs, vs);
    let vp7 = pos + vec3<f32>(0.0, vs, vs);

    if ((edges & 1u) != 0u)   { vertlist[0] = vertex_interp(vp0, v0, vp1, v1); }
    if ((edges & 2u) != 0u)   { vertlist[1] = vertex_interp(vp1, v1, vp2, v2); }
    if ((edges & 4u) != 0u)   { vertlist[2] = vertex_interp(vp2, v2, vp3, v3); }
    if ((edges & 8u) != 0u)   { vertlist[3] = vertex_interp(vp3, v3, vp0, v0); }
    if ((edges & 16u) != 0u)  { vertlist[4] = vertex_interp(vp4, v4, vp5, v5); }
    if ((edges & 32u) != 0u)  { vertlist[5] = vertex_interp(vp5, v5, vp6, v6); }
    if ((edges & 64u) != 0u)  { vertlist[6] = vertex_interp(vp6, v6, vp7, v7); }
    if ((edges & 128u) != 0u) { vertlist[7] = vertex_interp(vp7, v7, vp4, v4); }
    if ((edges & 256u) != 0u) { vertlist[8] = vertex_interp(vp0, v0, vp4, v4); }
    if ((edges & 512u) != 0u) { vertlist[9] = vertex_interp(vp1, v1, vp5, v5); }
    if ((edges & 1024u) != 0u) { vertlist[10] = vertex_interp(vp2, v2, vp6, v6); }
    if ((edges & 2048u) != 0u) { vertlist[11] = vertex_interp(vp3, v3, vp7, v7); }

    var i = 0u;
    loop {
        let t_idx = tables[tri_table_ptr + cube_index * 16u + i];
        if (t_idx == -1) { break; }

        let idx0 = tables[tri_table_ptr + cube_index * 16u + i];
        let idx1 = tables[tri_table_ptr + cube_index * 16u + i + 1u];
        let idx2 = tables[tri_table_ptr + cube_index * 16u + i + 2u];

        let p0_tri = vertlist[u32(idx0)];
        let p1_tri = vertlist[u32(idx1)];
        let p2_tri = vertlist[u32(idx2)];
        
        let n = normalize(cross(p1_tri - p0_tri, p2_tri - p0_tri));

        let current_count = atomicAdd(&counter, 3u);
        if (current_count + 3u < params.max_triangles * 3u) {
            vertices[current_count] = Vertex(vec4<f32>(p0_tri, 1.0), vec4<f32>(n, 0.0));
            vertices[current_count + 1u] = Vertex(vec4<f32>(p1_tri, 1.0), vec4<f32>(n, 0.0));
            vertices[current_count + 2u] = Vertex(vec4<f32>(p2_tri, 1.0), vec4<f32>(n, 0.0));
        }

        i += 3u;
    }
}
