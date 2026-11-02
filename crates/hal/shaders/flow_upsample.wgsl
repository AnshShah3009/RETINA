// Upsamples a dense flow field to the next finer pyramid level.
//
// Layout: planar CHW, like every multichannel tensor in the workspace.
// Plane 0 (`[0, src_w * src_h)`) is the horizontal flow u, plane 1 the
// vertical flow v. The output uses the same layout at the destination size.
//
// Sampling is OpenCV INTER_LINEAR: half-pixel centres, edge samples clamped.
// A displacement measured in coarse pixels is scaled to fine pixels, u by
// dst_w / src_w and v by dst_h / src_h (2x for a half-resolution level).
//
// The uniform is `ResizeParams` from `gpu_kernels/resize.rs`.

struct UpsampleParams {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    channels: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input_flow: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_flow: array<f32>;
@group(0) @binding(2) var<uniform> params: UpsampleParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.dst_w || y >= params.dst_h || params.src_w == 0u || params.src_h == 0u) {
        return;
    }

    let src_w_f = f32(params.src_w);
    let src_h_f = f32(params.src_h);
    let dst_w_f = f32(params.dst_w);
    let dst_h_f = f32(params.dst_h);

    let sx = max((f32(x) + 0.5) * src_w_f / dst_w_f - 0.5, 0.0);
    let sy = max((f32(y) + 0.5) * src_h_f / dst_h_f - 0.5, 0.0);

    let x0 = min(params.src_w - 1u, u32(floor(sx)));
    let y0 = min(params.src_h - 1u, u32(floor(sy)));
    let x1 = min(params.src_w - 1u, x0 + 1u);
    let y1 = min(params.src_h - 1u, y0 + 1u);
    let fx = select(0.0, clamp(sx - f32(x0), 0.0, 1.0), x0 != x1);
    let fy = select(0.0, clamp(sy - f32(y0), 0.0, 1.0), y0 != y1);

    let src_plane = params.src_w * params.src_h;
    let i00 = y0 * params.src_w + x0;
    let i10 = y0 * params.src_w + x1;
    let i01 = y1 * params.src_w + x0;
    let i11 = y1 * params.src_w + x1;

    let u = mix(
        mix(input_flow[i00], input_flow[i10], fx),
        mix(input_flow[i01], input_flow[i11], fx),
        fy
    );
    let v = mix(
        mix(input_flow[src_plane + i00], input_flow[src_plane + i10], fx),
        mix(input_flow[src_plane + i01], input_flow[src_plane + i11], fx),
        fy
    );

    let dst_plane = params.dst_w * params.dst_h;
    let o = y * params.dst_w + x;
    output_flow[o] = u * (dst_w_f / src_w_f);
    output_flow[dst_plane + o] = v * (dst_h_f / src_h_f);
}
