@group(0) @binding(0) var<storage, read> frame: array<f32>;
@group(0) @binding(1) var<storage, read_write> model: array<f32>;
@group(0) @binding(2) var<storage, read_write> mask: array<u32>;

struct Params {
    width: u32,
    height: u32,
    n_mixtures: u32,
    alpha: f32,
    var_threshold: f32,
    background_ratio: f32,
    var_init: f32,
    var_min: f32,
    var_max: f32,
}

@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let pixel_idx = y * params.width + x;
    let pixel = frame[pixel_idx];
    let pix_model_base = pixel_idx * params.n_mixtures * 3u;
    
    var fit_idx: i32 = -1;
    var foreground: bool = true;
    var total_weight: f32 = 0.0;

    // 1. Find Match
    for (var m: u32 = 0u; m < params.n_mixtures; m++) {
        let m_base = pix_model_base + m * 3u;
        let weight = model[m_base + 0u];
        let mean = model[m_base + 1u];
        let var_val = model[m_base + 2u];

        if (weight < 1e-5) { continue; }

        let diff = pixel - mean;
        let dist_sq = diff * diff;
        
        if (dist_sq < params.var_threshold * var_val) {
            fit_idx = i32(m);
            if (total_weight < params.background_ratio) {
                foreground = false;
            }
            break;
        }
        total_weight += weight;
    }

    mask[pixel_idx] = select(0u, 255u, foreground);

    // 2. Update Model
    if (fit_idx != -1) {
        for (var m: u32 = 0u; m < params.n_mixtures; m++) {
            let m_base = pix_model_base + m * 3u;
            if (i32(m) == fit_idx) {
                let w_val = model[m_base + 0u];
                let alpha_m = params.alpha / max(w_val, 1e-5);
                model[m_base + 0u] += params.alpha * (1.0 - w_val);
                
                let diff = pixel - model[m_base + 1u];
                model[m_base + 1u] += alpha_m * diff;
                let new_var = model[m_base + 2u] + alpha_m * (diff * diff - model[m_base + 2u]);
                model[m_base + 2u] = clamp(new_var, params.var_min, params.var_max);
            } else {
                model[m_base + 0u] *= (1.0 - params.alpha);
            }
        }
    } else {
        // No match: replace weakest
        var min_w_idx: u32 = 0u;
        var min_w: f32 = 2.0;
        for (var m: u32 = 0u; m < params.n_mixtures; m++) {
            let w = model[pix_model_base + m * 3u];
            if (w < min_w) {
                min_w = w;
                min_w_idx = m;
            }
        }
        
        let m_base = pix_model_base + min_w_idx * 3u;
        model[m_base + 0u] = params.alpha;
        model[m_base + 1u] = pixel;
        model[m_base + 2u] = params.var_init;
    }
}
