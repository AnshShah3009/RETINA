// Marching Cubes Pass 1: Count triangles per voxel
// Writes triangle count to counts buffer — no atomics needed.

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

@group(0) @binding(0) var<storage, read> voxels: array<Voxel>;
@group(0) @binding(1) var<storage, read_write> counts: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> tri_count_table: array<u32>; // 256 entries

fn get_val(x: u32, y: u32, z: u32) -> f32 {
    let idx = z * params.vol_x * params.vol_y + y * params.vol_x + x;
    return voxels[idx].tsdf;
}

// True when all eight corners of the cell at (x, y, z) have been observed.
// An unobserved voxel (weight 0) holds a placeholder, not a distance; meshing
// a sign change against it emitted a phantom back surface wherever the band
// behind a wall met the never-observed voxels further behind it.
fn cell_observed(x: u32, y: u32, z: u32) -> bool {
    let sx = params.vol_x;
    let sxy = params.vol_x * params.vol_y;
    let i = z * sxy + y * sx + x;
    return voxels[i].weight > 0.0
        && voxels[i + 1u].weight > 0.0
        && voxels[i + 1u + sx].weight > 0.0
        && voxels[i + sx].weight > 0.0
        && voxels[i + sxy].weight > 0.0
        && voxels[i + sxy + 1u].weight > 0.0
        && voxels[i + sxy + 1u + sx].weight > 0.0
        && voxels[i + sxy + sx].weight > 0.0;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let z = gid.z;

    let voxel_idx = z * (params.vol_x - 1u) * (params.vol_y - 1u) + y * (params.vol_x - 1u) + x;

    if (x >= params.vol_x - 1u || y >= params.vol_y - 1u || z >= params.vol_z - 1u) {
        return;
    }

    if (!cell_observed(x, y, z)) {
        counts[voxel_idx] = 0u;
        return;
    }

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

    counts[voxel_idx] = tri_count_table[cube_index];
}
