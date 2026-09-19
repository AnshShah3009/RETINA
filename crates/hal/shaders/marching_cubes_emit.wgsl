// Marching Cubes Pass 2: Emit vertices at exact offsets (no atomics)
// Reads prefix-summed offsets to know where to write each voxel's triangles.

struct Params {
    vol_x: u32,
    vol_y: u32,
    vol_z: u32,
    voxel_size: f32,
    iso_level: f32,
    _pad: u32,
}

struct Voxel {
    tsdf: f32,
    weight: f32,
}

struct Vertex {
    pos: vec4<f32>,
    norm: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> voxels: array<Voxel>;
@group(0) @binding(1) var<storage, read_write> vertices: array<Vertex>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> tables: array<i32>; // edge_table[256] + tri_table[4096]
@group(0) @binding(4) var<storage, read> offsets: array<u32>; // prefix-summed triangle offsets

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
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let z = gid.z;

    if (x >= params.vol_x - 1u || y >= params.vol_y - 1u || z >= params.vol_z - 1u) {
        return;
    }

    let voxel_idx = z * (params.vol_x - 1u) * (params.vol_y - 1u) + y * (params.vol_x - 1u) + x;
    let tri_offset = offsets[voxel_idx]; // where to write this voxel's triangles

    let v0 = get_val(x, y, z);
    let v1 = get_val(x + 1u, y, z);
    let v2 = get_val(x + 1u, y + 1u, z);
    let v3 = get_val(x, y + 1u, z);
    let v4 = get_val(x, y, z + 1u);
    let v5 = get_val(x + 1u, y, z + 1u);
    let v6 = get_val(x + 1u, y + 1u, z + 1u);
    let v7 = get_val(x, y + 1u, z + 1u);

    var cube_index = 0u;
    if (v0 < params.iso_level) { cube_index |= 1u; }
    if (v1 < params.iso_level) { cube_index |= 2u; }
    if (v2 < params.iso_level) { cube_index |= 4u; }
    if (v3 < params.iso_level) { cube_index |= 8u; }
    if (v4 < params.iso_level) { cube_index |= 16u; }
    if (v5 < params.iso_level) { cube_index |= 32u; }
    if (v6 < params.iso_level) { cube_index |= 64u; }
    if (v7 < params.iso_level) { cube_index |= 128u; }

    let edge_table_ptr = 0u;
    let tri_table_ptr = 256u;

    let edges = u32(tables[edge_table_ptr + cube_index]);
    if (edges == 0u) { return; }

    var vertlist: array<vec3<f32>, 12>;
    let vs = params.voxel_size;
    let pos = vec3<f32>(f32(x) * vs, f32(y) * vs, f32(z) * vs);

    let vp0 = pos;
    let vp1 = pos + vec3<f32>(vs, 0.0, 0.0);
    let vp2 = pos + vec3<f32>(vs, vs, 0.0);
    let vp3 = pos + vec3<f32>(0.0, vs, 0.0);
    let vp4 = pos + vec3<f32>(0.0, 0.0, vs);
    let vp5 = pos + vec3<f32>(vs, 0.0, vs);
    let vp6 = pos + vec3<f32>(vs, vs, vs);
    let vp7 = pos + vec3<f32>(0.0, vs, vs);

    if ((edges & 1u) != 0u)    { vertlist[0]  = vertex_interp(vp0, v0, vp1, v1); }
    if ((edges & 2u) != 0u)    { vertlist[1]  = vertex_interp(vp1, v1, vp2, v2); }
    if ((edges & 4u) != 0u)    { vertlist[2]  = vertex_interp(vp2, v2, vp3, v3); }
    if ((edges & 8u) != 0u)    { vertlist[3]  = vertex_interp(vp3, v3, vp0, v0); }
    if ((edges & 16u) != 0u)   { vertlist[4]  = vertex_interp(vp4, v4, vp5, v5); }
    if ((edges & 32u) != 0u)   { vertlist[5]  = vertex_interp(vp5, v5, vp6, v6); }
    if ((edges & 64u) != 0u)   { vertlist[6]  = vertex_interp(vp6, v6, vp7, v7); }
    if ((edges & 128u) != 0u)  { vertlist[7]  = vertex_interp(vp7, v7, vp4, v4); }
    if ((edges & 256u) != 0u)  { vertlist[8]  = vertex_interp(vp0, v0, vp4, v4); }
    if ((edges & 512u) != 0u)  { vertlist[9]  = vertex_interp(vp1, v1, vp5, v5); }
    if ((edges & 1024u) != 0u) { vertlist[10] = vertex_interp(vp2, v2, vp6, v6); }
    if ((edges & 2048u) != 0u) { vertlist[11] = vertex_interp(vp3, v3, vp7, v7); }

    // Emit triangles at exact offset — no atomics
    var vert_write = tri_offset * 3u; // 3 vertices per triangle
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

        vertices[vert_write]      = Vertex(vec4<f32>(p0_tri, 1.0), vec4<f32>(n, 0.0));
        vertices[vert_write + 1u] = Vertex(vec4<f32>(p1_tri, 1.0), vec4<f32>(n, 0.0));
        vertices[vert_write + 2u] = Vertex(vec4<f32>(p2_tri, 1.0), vec4<f32>(n, 0.0));

        vert_write += 3u;
        i += 3u;
    }
}
