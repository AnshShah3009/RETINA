struct Params {
    width: u32,
    height: u32,
    num_rho: u32,
    num_theta: u32,
    rho_res: f32,
    theta_res: f32,
    threshold: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> accumulator: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: Params;

// Pass 1: Accumulator Voting
@compute @workgroup_size(16, 16)
fn vote(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let idx = y * params.width + x;
    let combined = input_data[idx / 4u];
    let pixel_val = (combined >> ((idx % 4u) * 8u)) & 0xFFu;

    if (pixel_val == 0u) {
        return;
    }

    // Edge pixel found, vote for all possible thetas
    for (var i = 0u; i < params.num_theta; i++) {
        let theta = f32(i) * params.theta_res;
        let rho = f32(x) * cos(theta) + f32(y) * sin(theta);
        
        // rho can be negative, offset by max possible rho / 2
        let rho_idx = i32(rho / params.rho_res + f32(params.num_rho) / 2.0);
        
        if (rho_idx >= 0 && u32(rho_idx) < params.num_rho) {
            let acc_idx = u32(rho_idx) * params.num_theta + i;
            atomicAdd(&accumulator[acc_idx], 1u);
        }
    }
}

// Pass 2: Peak detection (finding local maxima in accumulator)
// This pass could output to a list of detected lines.
// For simplicity, we'll return the accumulator and do final peak extraction on CPU for now, 
// or implement a simple suppression here.
