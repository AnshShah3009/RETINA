// Marker Detection GPU Compute Shader
// Samples grid bits from candidate bounding boxes and decodes marker payloads

struct Candidate {
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
    grid_size: u32,
    payload_bits: u32,
}

struct MarkerResult {
    bitmask: u32,          // Lower 32 bits of payload (up to 6x6 = 36 bits)
    bitmask_high: u32,     // Upper 32 bits for larger payloads
    best_id: u32,
    rotation: u32,
    confidence: f32,
    status: u32,           // 0 = invalid, 1 = valid, 2 = border_fail, 3 = no_match
}

struct Params {
    image_width: u32,
    image_height: u32,
    num_candidates: u32,
    dict_size: u32,
    border_bits: u32,
    threshold: u32,        // Intensity threshold (default 128)
    max_hamming: u32,      // Maximum allowed hamming distance for match
    _padding: u32,
}

@group(0) @binding(0) var image_texture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read> candidates: array<Candidate>;
@group(0) @binding(2) var<storage, read_write> results: array<MarkerResult>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> dictionary: array<u32>; // Pairs of (low, high) for each code

// Sample a single pixel from the texture (clamped)
fn sample_pixel(x: i32, y: i32) -> f32 {
    let cx = clamp(x, 0, i32(params.image_width) - 1);
    let cy = clamp(y, 0, i32(params.image_height) - 1);
    let val = textureLoad(image_texture, vec2<i32>(cx, cy), 0);
    return val.r;
}

// Sample a grid cell and determine if it's black (1) or white (0)
fn sample_grid_cell(
    min_x: f32, min_y: f32,
    bw: f32, bh: f32,
    gx: u32, gy: u32,
    grid: u32
) -> u32 {
    let x0 = min_x + f32(gx) * bw / f32(grid);
    let x1 = min_x + f32(gx + 1u) * bw / f32(grid);
    let y0 = min_y + f32(gy) * bh / f32(grid);
    let y1 = min_y + f32(gy + 1u) * bh / f32(grid);
    
    // Sample center of cell for simplicity (bilinear would be better but more complex)
    let cx = (x0 + x1) * 0.5;
    let cy = (y0 + y1) * 0.5;
    
    // Sample 3x3 grid within cell for robustness
    var sum: f32 = 0.0;
    var count: f32 = 0.0;
    
    let cell_w = (x1 - x0) * 0.25;
    let cell_h = (y1 - y0) * 0.25;
    
    for (var dy: i32 = -1; dy <= 1; dy++) {
        for (var dx: i32 = -1; dx <= 1; dx++) {
            let sx = i32(cx + f32(dx) * cell_w);
            let sy = i32(cy + f32(dy) * cell_h);
            sum += sample_pixel(sx, sy);
            count += 1.0;
        }
    }
    
    let avg = sum / count;
    let threshold_f = f32(params.threshold) / 255.0;
    
    // Return 1 if black (below threshold), 0 if white
    return select(0u, 1u, avg < threshold_f);
}

// Check if all border cells are black
fn check_border(bits: ptr<function, array<u32, 64>>, grid: u32) -> bool {
    // Top and bottom rows
    for (var i: u32 = 0u; i < grid; i++) {
        if ((*bits)[i] == 0u) { return false; }
        if ((*bits)[(grid - 1u) * grid + i] == 0u) { return false; }
    }
    // Left and right columns (excluding corners already checked)
    for (var i: u32 = 1u; i < grid - 1u; i++) {
        if ((*bits)[i * grid] == 0u) { return false; }
        if ((*bits)[i * grid + (grid - 1u)] == 0u) { return false; }
    }
    return true;
}

// Extract payload bits from grid (excluding border)
fn extract_payload(
    bits: ptr<function, array<u32, 64>>,
    payload_bits: u32,
    border_bits: u32,
    grid: u32
) -> vec2<u32> {
    var code_low: u32 = 0u;
    var code_high: u32 = 0u;
    
    for (var y: u32 = 0u; y < payload_bits; y++) {
        for (var x: u32 = 0u; x < payload_bits; x++) {
            let v = (*bits)[(y + border_bits) * grid + (x + border_bits)];
            if (v != 0u) {
                let idx = y * payload_bits + x;
                if (idx < 32u) {
                    code_low |= (1u << idx);
                } else {
                    code_high |= (1u << (idx - 32u));
                }
            }
        }
    }
    
    return vec2<u32>(code_low, code_high);
}

// Rotate code 90 degrees clockwise
fn rotate_code_90(code_low: u32, code_high: u32, side: u32) -> vec2<u32> {
    var out_low: u32 = 0u;
    var out_high: u32 = 0u;
    
    for (var y: u32 = 0u; y < side; y++) {
        for (var x: u32 = 0u; x < side; x++) {
            let idx = y * side + x;
            var bit: u32 = 0u;
            if (idx < 32u) {
                bit = (code_low >> idx) & 1u;
            } else {
                bit = (code_high >> (idx - 32u)) & 1u;
            }
            
            if (bit != 0u) {
                let nx = side - 1u - y;
                let ny = x;
                let nidx = ny * side + nx;
                if (nidx < 32u) {
                    out_low |= (1u << nidx);
                } else {
                    out_high |= (1u << (nidx - 32u));
                }
            }
        }
    }
    
    return vec2<u32>(out_low, out_high);
}

// Count differing bits (hamming distance)
fn hamming_distance(a_low: u32, a_high: u32, b_low: u32, b_high: u32) -> u32 {
    return countOneBits(a_low ^ b_low) + countOneBits(a_high ^ b_high);
}

// Main compute shader - one workgroup per candidate
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let candidate_idx = global_id.x;
    
    if (candidate_idx >= params.num_candidates) {
        return;
    }
    
    let candidate = candidates[candidate_idx];
    let grid = candidate.grid_size;
    let payload_bits = candidate.payload_bits;
    let border_bits = params.border_bits;
    
    // Bounds check
    if (grid > 8u || grid == 0u) {
        results[candidate_idx].status = 0u;
        return;
    }
    
    let min_x = f32(candidate.min_x);
    let min_y = f32(candidate.min_y);
    let bw = f32(candidate.max_x - candidate.min_x + 1u);
    let bh = f32(candidate.max_y - candidate.min_y + 1u);
    
    // Sample all grid cells
    var bits: array<u32, 64>; // Max 8x8 grid
    for (var gy: u32 = 0u; gy < grid; gy++) {
        for (var gx: u32 = 0u; gx < grid; gx++) {
            bits[gy * grid + gx] = sample_grid_cell(min_x, min_y, bw, bh, gx, gy, grid);
        }
    }
    
    // Check border
    if (!check_border(&bits, grid)) {
        results[candidate_idx].status = 2u; // border_fail
        results[candidate_idx].confidence = 0.0;
        return;
    }
    
    // Extract payload
    let payload = extract_payload(&bits, payload_bits, border_bits, grid);
    
    // Generate all 4 rotations
    var rots: array<vec2<u32>, 4>;
    rots[0] = payload;
    rots[1] = rotate_code_90(rots[0].x, rots[0].y, payload_bits);
    rots[2] = rotate_code_90(rots[1].x, rots[1].y, payload_bits);
    rots[3] = rotate_code_90(rots[2].x, rots[2].y, payload_bits);
    
    // Find best match in dictionary
    var best_id: u32 = 0u;
    var best_rot: u32 = 0u;
    var best_dist: u32 = 0xFFFFFFFFu;
    
    for (var id: u32 = 0u; id < params.dict_size; id++) {
        let dict_low = dictionary[id * 2u];
        let dict_high = dictionary[id * 2u + 1u];
        
        for (var rot: u32 = 0u; rot < 4u; rot++) {
            let dist = hamming_distance(rots[rot].x, rots[rot].y, dict_low, dict_high);
            if (dist < best_dist) {
                best_dist = dist;
                best_id = id;
                best_rot = rot;
            }
        }
    }
    
    // Store result
    results[candidate_idx].bitmask = payload.x;
    results[candidate_idx].bitmask_high = payload.y;
    results[candidate_idx].best_id = best_id;
    results[candidate_idx].rotation = best_rot;
    
    // Compute confidence based on hamming distance
    let total_bits = payload_bits * payload_bits;
    let confidence = 1.0 - f32(best_dist) / f32(total_bits);
    results[candidate_idx].confidence = confidence;
    
    // Check if match is valid
    if (best_dist <= params.max_hamming) {
        results[candidate_idx].status = 1u; // valid
    } else {
        results[candidate_idx].status = 3u; // no_match
    }
}
