struct Params {
    img_w: u32,
    img_h: u32,
    templ_w: u32,
    templ_h: u32,
    method: u32, 
}

@group(0) @binding(0) var<storage, read> img_data: array<u32>;
@group(0) @binding(1) var<storage, read> templ_data: array<u32>;
@group(0) @binding(2) var<storage, read_write> score_map: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

fn get_img_pixel(x: u32, y: u32) -> f32 {
    let idx = y * params.img_w + x;
    let combined = img_data[idx / 4u];
    return f32((combined >> ((idx % 4u) * 8u)) & 0xFFu);
}

fn get_templ_pixel(x: u32, y: u32) -> f32 {
    let idx = y * params.templ_w + x;
    let combined = templ_data[idx / 4u];
    return f32((combined >> ((idx % 4u) * 8u)) & 0xFFu);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    let out_w = params.img_w - params.templ_w + 1u;
    let out_h = params.img_h - params.templ_h + 1u;

    if (x >= out_w || y >= out_h) {
        return;
    }

    var score = 0.0;
    
    // method: 0: SqDiff, 1: SqDiffNormed, 2: Ccorr, 3: CcorrNormed, 4: Ccoeff, 5: CcoeffNormed
    if (params.method == 0u) {
        for (var j = 0u; j < params.templ_h; j = j + 1u) {
            for (var i = 0u; i < params.templ_w; i = i + 1u) {
                let iv = get_img_pixel(x + i, y + j);
                let tv = get_templ_pixel(i, j);
                let diff = iv - tv;
                score = score + diff * diff;
            }
        }
    } else if (params.method == 2u) {
        for (var j = 0u; j < params.templ_h; j = j + 1u) {
            for (var i = 0u; i < params.templ_w; i = i + 1u) {
                let iv = get_img_pixel(x + i, y + j);
                let tv = get_templ_pixel(i, j);
                score = score + iv * tv;
            }
        }
    } else {
        // Fallback for other methods (not yet optimized for GPU)
        score = 0.0;
    }
    
    score_map[y * out_w + x] = score;
}
