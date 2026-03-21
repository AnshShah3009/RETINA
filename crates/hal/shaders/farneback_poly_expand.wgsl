// Polynomial Expansion Kernel for Farneback Optical Flow
// Reference: OpenCV optflowgf.cpp lines 116-203
// Fits quadratic polynomial to local neighborhoods: I(x) ≈ a*x² + b*x*y + c*y² + d*x + e*y + f

struct PolyExpandParams {
    src_w: u32,
    src_h: u32,
    stride: u32,
    poly_n: u32,     // Polynomial degree (5 or 7)
    sigma: f32,       // Gaussian sigma for weighting
}

// Gaussian weight function
fn gaussian_weight(x: f32, sigma: f32) -> f32 {
    let two_sigma_sq = 2.0 * sigma * sigma;
    return exp(-(x * x) / two_sigma_sq);
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: PolyExpandParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.src_w || y >= params.src_h) {
        return;
    }
    
    let idx = y * params.stride + x;
    let center_val = input[idx];
    
    // For polyN=5, we compute: a, b, c, d, e, f (6 coefficients)
    // For polyN=7, we compute: g, h, k (additional higher-order terms)
    
    // Polynomial coefficients stored as: [f, d, e, a, b, c] per pixel
    // This is the order expected by the flow estimation step
    
    var sum_f = 0.0;
    var sum_d = 0.0;  // d coeff (gradient in x)
    var sum_e = 0.0;  // e coeff (gradient in y)
    var sum_a = 0.0;  // a coeff (x²)
    var sum_b = 0.0;  // b coeff (xy)
    var sum_c = 0.0;  // c coeff (y²)
    var weight_sum = 0.0;
    
    let half_win = params.poly_n / 2u;
    
    for (var dy = 0u; dy < params.poly_n; dy = dy + 1u) {
        for (var dx = 0u; dx < params.poly_n; dx = dx + 1u) {
            let sx = i32(x) + i32(dx) - i32(half_win);
            let sy = i32(y) + i32(dy) - i32(half_win);
            
            // Boundary check
            if (sx < 0 || sx >= i32(params.src_w) || sy < 0 || sy >= i32(params.src_h)) {
                continue;
            }
            
            let src_idx = u32(sy) * params.stride + u32(sx);
            let val = input[src_idx];
            let diff = val - center_val;
            
            // Compute Gaussian weight
            let fx = f32(i32(dx) - i32(half_win));
            let fy = f32(i32(dy) - i32(half_win));
            let w = gaussian_weight(fx, params.sigma) * gaussian_weight(fy, params.sigma);
            
            // Accumulate polynomial coefficients
            sum_f += w * diff;
            sum_d += w * diff * fx;
            sum_e += w * diff * fy;
            sum_a += w * fx * fx;
            sum_b += w * fx * fy;
            sum_c += w * fy * fy;
            weight_sum += w;
        }
    }
    
    if (weight_sum > 0.0001) {
        let inv_w = 1.0 / weight_sum;
        output[idx * 6u + 0u] = sum_f * inv_w;  // f (constant term)
        output[idx * 6u + 1u] = sum_d * inv_w;  // d (gradient x)
        output[idx * 6u + 2u] = sum_e * inv_w;  // e (gradient y)
        output[idx * 6u + 3u] = sum_a * inv_w;  // a (x²)
        output[idx * 6u + 4u] = sum_b * inv_w;  // b (xy)
        output[idx * 6u + 5u] = sum_c * inv_w;  // c (y²)
    } else {
        output[idx * 6u + 0u] = 0.0;
        output[idx * 6u + 1u] = 0.0;
        output[idx * 6u + 2u] = 0.0;
        output[idx * 6u + 3u] = 0.0;
        output[idx * 6u + 4u] = 0.0;
        output[idx * 6u + 5u] = 0.0;
    }
}
