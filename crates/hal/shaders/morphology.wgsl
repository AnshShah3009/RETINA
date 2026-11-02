// GPU morphology (erode / dilate) on packed u8 images.
//
// Matches the CPU backend (`cpu/compute_context_impl.rs`, `morphology`) bit
// for bit:
//
// * The anchor is the kernel centre `(kw / 2, kh / 2)`, so an even-sized or
//   one-sided mask samples off-centre exactly as on the CPU.
// * The border is replicate: an out-of-image tap reads the nearest edge pixel.
//   The morphological identity (255 for erode, 0 for dilate) only seeds the
//   reduction, so it is the result only when the mask has no live taps.
// * A tensor with several channels is a stack of `len / (width * height)`
//   planes (CHW), each eroded or dilated on its own.
//
// The image and the mask are both u8 tensors resident on the GPU, packed four
// bytes per word. Each invocation writes one output word, that is four
// consecutive bytes of the dense buffer, so a row whose width is not a
// multiple of four and a plane that starts mid-word both work.

struct Params {
    width: u32,
    height: u32,
    kw: u32,
    kh: u32,
    typ: u32, // 0: Erode, 1: Dilate
    len: u32, // pixels in the whole tensor, all planes
}

@group(0) @binding(0) var<storage, read> input_data: array<u32>; // packed u8
@group(0) @binding(1) var<storage, read> kernel_mask: array<u32>; // packed u8, nonzero = live tap
@group(0) @binding(2) var<storage, read_write> output_data: array<u32>; // packed u8
@group(0) @binding(3) var<uniform> params: Params;

fn byte_at_input(i: u32) -> u32 {
    return (input_data[i >> 2u] >> ((i & 3u) << 3u)) & 0xFFu;
}

fn mask_at(i: u32) -> u32 {
    return (kernel_mask[i >> 2u] >> ((i & 3u) << 3u)) & 0xFFu;
}

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    // A 1-D range dispatched as a 2-D grid so no axis exceeds 65535
    // workgroups (see gpu_kernels::dispatch_1d).
    let word = global_id.x + global_id.y * num_wg.x * 64u;
    // An empty axis has no pixel to replicate. The host never dispatches for
    // one, and this guard keeps the divisions below nonzero for a raw
    // `ComputeContext::dispatch` uniform too.
    if (params.width == 0u || params.height == 0u) {
        return;
    }
    if (word * 4u >= params.len) {
        return;
    }

    let w = params.width;
    let plane_size = w * params.height;
    let cx = i32(params.kw / 2u);
    let cy = i32(params.kh / 2u);
    let max_x = i32(w) - 1;
    let max_y = i32(params.height) - 1;
    let is_erode = params.typ == 0u;

    var packed = 0u;
    for (var j = 0u; j < 4u; j++) {
        let idx = word * 4u + j;
        if (idx >= params.len) {
            break;
        }
        // `plane_size` is the product of two nonzero values. It can only be 0
        // if it wraps, and then no pixel of the tensor fits in a u32 index.
        let plane = idx / max(plane_size, 1u);
        let rem = idx - plane * plane_size;
        let x = i32(rem % w);
        let y = i32(rem / w);
        let base = plane * plane_size;

        var val = 0u;
        if (is_erode) {
            val = 255u;
        }
        for (var ky = 0u; ky < params.kh; ky++) {
            let sy = u32(clamp(y + i32(ky) - cy, 0, max_y));
            for (var kx = 0u; kx < params.kw; kx++) {
                if (mask_at(ky * params.kw + kx) == 0u) {
                    continue;
                }
                let sx = u32(clamp(x + i32(kx) - cx, 0, max_x));
                let pixel = byte_at_input(base + sy * w + sx);
                if (is_erode) {
                    val = min(val, pixel);
                } else {
                    val = max(val, pixel);
                }
            }
        }
        packed = packed | (val << (j * 8u));
    }

    output_data[word] = packed;
}
