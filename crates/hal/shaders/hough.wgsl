// Standard Hough lines: accumulator voting with exact CPU-equivalent binning.
//
// The host uploads the angle tables already scaled by 1/rho, so the bin of
// pixel (x, y) at angle n is round(x * c[n] + y * s[n]) + rho_offset, theta-major
// (`n * num_rho + r`) - OpenCV's `HoughLinesStandard` layout. The CPU paths
// round an f32 expression; this shader computes (nearly) the same f32 value
// but a driver may fuse `x * c + y * s` into an FMA, and the HAL's convention
// divides by rho after the sum. Both differ from the CPU value by a few ulps
// of |x c| + |y s|, which only matters when the value lies that close to a
// rounding boundary k + 0.5. Such votes are not cast here: they are appended
// to `ambiguous` as (pixel, angle) pairs and the host casts them with the
// exact CPU formula. Every other vote rounds the same way as on the CPU, so
// the accumulator is bit-identical.
//
// Pixels are indexed linearly over a 2-D grid of workgroups (see
// gpu_kernels::dispatch_1d) so no dispatch axis exceeds 65535.

struct Params {
    width: u32,
    height: u32,
    num_rho: u32,
    num_angle: u32,
    rho_offset: i32,
    ambiguous_capacity: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
// (cos(theta_n) / rho, sin(theta_n) / rho)
@group(0) @binding(1) var<storage, read> trig: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> accumulator: array<atomic<u32>>;
// [0] = number of ambiguous votes (may exceed capacity), then pairs.
@group(0) @binding(3) var<storage, read_write> ambiguous: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params: Params;

// Generous bound on |GPU value - CPU value| relative to |x c| + |y s|; the
// actual difference is at most ~4 ulp (2^-22).
const AMBIGUITY: f32 = 9.5367431640625e-7; // 2^-20

@compute @workgroup_size(256)
fn vote(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let i = gid.x + gid.y * num_wg.x * 256u;
    if (i >= params.width * params.height) {
        return;
    }
    // Nonzero positive pixels vote (NaN does not), as on the CPU.
    if (!(input_data[i] > 0.0)) {
        return;
    }
    let x = f32(i % params.width);
    let y = f32(i / params.width);

    for (var n = 0u; n < params.num_angle; n++) {
        let t = trig[n];
        let a = x * t.x;
        let b = y * t.y;
        let r = a + b;
        let fr = floor(r);
        let frac = r - fr; // exact
        if (abs(frac - 0.5) <= (abs(a) + abs(b)) * AMBIGUITY) {
            let slot = atomicAdd(&ambiguous[0], 1u);
            if (slot < params.ambiguous_capacity) {
                atomicStore(&ambiguous[1u + 2u * slot], i);
                atomicStore(&ambiguous[2u + 2u * slot], n);
            }
            continue;
        }
        let k = i32(fr) + select(0, 1, frac > 0.5) + params.rho_offset;
        if (k >= 0 && u32(k) < params.num_rho) {
            atomicAdd(&accumulator[n * params.num_rho + u32(k)], 1u);
        }
    }
}
