// Canny, bit-exact with `cv::Canny` (aperture 3) on an 8-bit image - the same
// integer algorithm as the CPU reference in `cpu/canny_opencv.rs`:
//
//   * input rounded (ties to even) and saturated to 0..=255, NaN -> 0;
//   * 3x3 Sobel with BORDER_REPLICATE, no pre-blur;
//   * magnitude |dx| + |dy| (L1) or dx² + dy² (L2, thresholds squared on the
//     host); pixels outside the image have magnitude 0;
//   * NMS with OpenCV's fixed-point tan(22.5°) test: strict towards left/up
//     (`>`), non-strict towards right/down (`>=`), strict on both diagonal
//     neighbours;
//   * thresholds strict: m > low is a candidate, m > high is strong.
//
// Everything is i32 arithmetic (|dx|,|dy| <= 1020, so dx² + dy² <= 2080800
// and |dy| << 15 <= 33423360 fit), hence exact on any GPU. The output is the
// NMS state map (0 none / 1 weak / 2 strong); the host runs the 8-connected
// hysteresis flood fill, shared with the CPU path.
//
// Both passes index pixels linearly over a 2-D grid of workgroups (see
// gpu_kernels::dispatch_1d) so no dispatch axis exceeds 65535.

struct Params {
    width: u32,
    height: u32,
    low: i32,
    high: i32,
    l2: u32,
    tg22: i32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(3) var<uniform> params: Params;

// ---- Pass 1: Sobel -----------------------------------------------------------

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
// dx in the low 16 bits, dy in the high 16 bits (two's complement).
@group(0) @binding(1) var<storage, read_write> grad_out: array<u32>;

fn pixel(x: i32, y: i32) -> i32 {
    let ix = clamp(x, 0, i32(params.width) - 1);
    let iy = clamp(y, 0, i32(params.height) - 1);
    let v = input_data[u32(iy) * params.width + u32(ix)];
    // NaN check on the bits: float comparisons may assume finite math.
    if ((bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u) {
        return 0;
    }
    return i32(clamp(round(v), 0.0, 255.0));
}

@compute @workgroup_size(256)
fn sobel(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let i = gid.x + gid.y * num_wg.x * 256u;
    if (i >= params.width * params.height) {
        return;
    }
    let x = i32(i % params.width);
    let y = i32(i / params.width);

    let p00 = pixel(x - 1, y - 1);
    let p01 = pixel(x, y - 1);
    let p02 = pixel(x + 1, y - 1);
    let p10 = pixel(x - 1, y);
    let p12 = pixel(x + 1, y);
    let p20 = pixel(x - 1, y + 1);
    let p21 = pixel(x, y + 1);
    let p22 = pixel(x + 1, y + 1);

    let dx = (p02 - p00) + 2 * (p12 - p10) + (p22 - p20);
    let dy = (p20 + 2 * p21 + p22) - (p00 + 2 * p01 + p02);
    grad_out[i] = (bitcast<u32>(dx) & 0xffffu) | (bitcast<u32>(dy) << 16u);
}

// ---- Pass 2: magnitude + NMS + double threshold ------------------------------

@group(0) @binding(1) var<storage, read> grad_in: array<u32>;
// Four 8-bit states per word, pixel 4k + j in byte j (little-endian bytes).
@group(0) @binding(2) var<storage, read_write> state_out: array<u32>;

fn grad_at(i: u32) -> vec2<i32> {
    let g = bitcast<i32>(grad_in[i]);
    return vec2<i32>((g << 16u) >> 16u, g >> 16u);
}

fn mag_of(g: vec2<i32>) -> i32 {
    if (params.l2 != 0u) {
        return g.x * g.x + g.y * g.y;
    }
    return abs(g.x) + abs(g.y);
}

fn mag_at(x: i32, y: i32) -> i32 {
    if (x < 0 || y < 0 || x >= i32(params.width) || y >= i32(params.height)) {
        return 0;
    }
    return mag_of(grad_at(u32(y) * params.width + u32(x)));
}

fn nms_state(i: u32) -> u32 {
    let g = grad_at(i);
    let m = mag_of(g);
    if (m <= params.low) {
        return 0u;
    }
    let x = i32(i % params.width);
    let y = i32(i / params.width);
    let ax = abs(g.x);
    let ay = abs(g.y) << 15u;
    let tg22x = ax * params.tg22;
    var is_max = false;
    if (ay < tg22x) {
        is_max = m > mag_at(x - 1, y) && m >= mag_at(x + 1, y);
    } else {
        let tg67x = tg22x + (ax << 16u);
        if (ay > tg67x) {
            is_max = m > mag_at(x, y - 1) && m >= mag_at(x, y + 1);
        } else {
            let s = select(1, -1, (g.x ^ g.y) < 0);
            is_max = m > mag_at(x - s, y - 1) && m > mag_at(x + s, y + 1);
        }
    }
    if (!is_max) {
        return 0u;
    }
    return select(1u, 2u, m > params.high);
}

@compute @workgroup_size(256)
fn nms(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let word = gid.x + gid.y * num_wg.x * 256u;
    let n = params.width * params.height;
    if (word * 4u >= n) {
        return;
    }
    var packed = 0u;
    for (var j = 0u; j < 4u; j++) {
        let i = word * 4u + j;
        if (i < n) {
            packed = packed | (nms_state(i) << (8u * j));
        }
    }
    state_out[word] = packed;
}
