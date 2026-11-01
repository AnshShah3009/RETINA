// OpenCV `pyrDown` on planar CHW f32.
// Output size is ((src_w + 1) / 2, (src_h + 1) / 2). Each output pixel is the
// 5×5 [1 4 6 4 1] filter centred on source (2x, 2y), BORDER_REFLECT_101,
// divided by 256.

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

fn reflect101(p: i32, len: i32) -> i32 {
    if (len <= 1) {
        return 0;
    }
    // Fold by reflection until in range, without `%`: the signed-remainder
    // form gave wrong taps for negative coordinates on this backend, so row 0
    // and column 0 disagreed with the CPU path and cv2.pyrDown (row 0 of a
    // 2x + y ramp read 15.6 instead of 2.25). The taps reach at most 2 past
    // an edge, so this loops at most a few times even for len 2.
    var c = p;
    loop {
        if (c < 0) {
            c = -c;
        } else if (c >= len) {
            c = 2 * len - 2 - c;
        } else {
            break;
        }
    }
    return c;
}

fn tap(i: i32) -> f32 {
    if (i == 2) {
        return 6.0;
    }
    if (i == 1 || i == 3) {
        return 4.0;
    }
    return 1.0;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    let ch = i32(gid.z);
    if (x >= i32(params.dst_w) || y >= i32(params.dst_h) || ch >= i32(params.channels)) {
        return;
    }
    let sw = i32(params.src_w);
    let sh = i32(params.src_h);
    let src_plane = ch * sh * sw;
    var acc = 0.0;
    for (var j = 0; j < 5; j++) {
        let sy = reflect101(2 * y + j - 2, sh);
        var row = 0.0;
        for (var i = 0; i < 5; i++) {
            let sx = reflect101(2 * x + i - 2, sw);
            row = row + tap(i) * input_data[src_plane + sy * sw + sx];
        }
        acc = acc + tap(j) * row;
    }
    let dst_plane = ch * i32(params.dst_h) * i32(params.dst_w);
    output_data[dst_plane + y * i32(params.dst_w) + x] = acc / 256.0;
}
