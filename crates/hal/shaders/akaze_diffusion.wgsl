// AKAZE Non-linear Diffusion Kernel
// Implements FED (Fast Explicit Diffusion) step

struct Params {
    width: u32,
    height: u32,
    k: f32,      // Contrast factor
    tau: f32,    // Time step
}

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

// Perona-Malik 2 diffusivity: g(x) = 1 / (1 + x^2/k^2)
fn diffusivity(grad_sq: f32) -> f32 {
    return 1.0 / (1.0 + grad_sq / (params.k * params.k));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    let w = i32(params.width);
    let h = i32(params.height);

    if (x >= w || y >= h) {
        return;
    }

    // Indices for 4-neighborhood
    let idx = y * w + x;
    let idx_n = max(y - 1, 0) * w + x;
    let idx_s = min(y + 1, h - 1) * w + x;
    let idx_w = y * w + max(x - 1, 0);
    let idx_e = y * w + min(x + 1, w - 1);

    let center = input_data[idx];
    let n = input_data[idx_n];
    let s = input_data[idx_s];
    let west = input_data[idx_w];
    let e = input_data[idx_e];

    // Gradients
    let grad_n = n - center;
    let grad_s = s - center;
    let grad_w = west - center;
    let grad_e = e - center;

    // Conductances
    let g_n = diffusivity(grad_n * grad_n);
    let g_s = diffusivity(grad_s * grad_s);
    let g_w = diffusivity(grad_w * grad_w);
    let g_e = diffusivity(grad_e * grad_e);

    // Diffusion update: L_next = L + tau * div(g * grad(L))
    output_data[idx] = center + params.tau * (g_n * grad_n + g_s * grad_s + g_w * grad_w + g_e * grad_e);
}
