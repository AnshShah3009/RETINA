// Bilinear resize of a planar CHW f32 tensor, half-pixel mapping.
//
// Element (ch, y, x) of a `TensorShape::new(c, h, w)` tensor is at
// `ch * h * w + y * w + x`. Each plane is an independent image; the channel is
// `global_id.z` (the host dispatches `channels` workgroups in z).

struct Params {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    channels: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x_dst = global_id.x;
    let y_dst = global_id.y;
    let ch = global_id.z;

    // An empty source has nothing to sample (and `src_w - 1u` would wrap);
    // the host returns zeros for it without dispatching. A raw `dispatch`
    // uniform with `channels == 0` predates the field being read: one plane.
    if (x_dst >= params.dst_w || y_dst >= params.dst_h || ch >= max(params.channels, 1u)
        || params.src_w == 0u || params.src_h == 0u) {
        return;
    }

    let src_width_f = f32(params.src_w);
    let src_height_f = f32(params.src_h);
    // Half-pixel mapping divides by the destination size, which is never zero
    // for a dispatched work item, so no degenerate-size guard is needed.
    let dst_width_f = max(f32(params.dst_w), 1.0);
    let dst_height_f = max(f32(params.dst_h), 1.0);

    // Lower-bound in float; upper-bound index clamp in integer space.
    let src_x_f = max((f32(x_dst) + 0.5) * src_width_f / dst_width_f - 0.5, 0.0);
    let src_y_f = max((f32(y_dst) + 0.5) * src_height_f / dst_height_f - 0.5, 0.0);

    let x0 = min(params.src_w - 1u, u32(floor(src_x_f)));
    let y0 = min(params.src_h - 1u, u32(floor(src_y_f)));
    let x1 = min(params.src_w - 1u, x0 + 1u);
    let y1 = min(params.src_h - 1u, y0 + 1u);

    let dx = select(0.0, clamp(src_x_f - f32(x0), 0.0, 1.0), x0 != x1);
    let dy = select(0.0, clamp(src_y_f - f32(y0), 0.0, 1.0), y0 != y1);

    let src_plane = ch * params.src_h * params.src_w;
    let p00 = input_data[src_plane + y0 * params.src_w + x0];
    let p10 = input_data[src_plane + y0 * params.src_w + x1];
    let p01 = input_data[src_plane + y1 * params.src_w + x0];
    let p11 = input_data[src_plane + y1 * params.src_w + x1];

    // The CPU backend's order: (p00 (1 - dx) + p10 dx) (1 - dy) + (...) dy.
    let v0 = p00 * (1.0 - dx) + p10 * dx;
    let v1 = p01 * (1.0 - dx) + p11 * dx;
    let dst_plane = ch * params.dst_h * params.dst_w;
    output_data[dst_plane + y_dst * params.dst_w + x_dst] = v0 * (1.0 - dy) + v1 * dy;
}
