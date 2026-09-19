struct Params {
    num_boxes: u32,
    threshold: f32,
}

@group(0) @binding(0) var<storage, read> boxes: array<f32>; // [x1, y1, x2, y2, score] * num_boxes
@group(0) @binding(1) var<storage, read_write> iou_matrix: array<u32>; // Bitmask? Or just bool? Let's use 1 u32 per pair.
@group(0) @binding(2) var<uniform> params: Params;

fn compute_iou(idx1: u32, idx2: u32) -> f32 {
    let b1_offset = idx1 * 5u;
    let b2_offset = idx2 * 5u;

    let x1 = max(boxes[b1_offset], boxes[b2_offset]);
    let y1 = max(boxes[b1_offset + 1u], boxes[b2_offset + 1u]);
    let x2 = min(boxes[b1_offset + 2u], boxes[b2_offset + 2u]);
    let y2 = min(boxes[b1_offset + 3u], boxes[b2_offset + 3u]);

    let w = max(0.0, x2 - x1);
    let h = max(0.0, y2 - y1);
    let inter = w * h;

    if (inter == 0.0) { return 0.0; }

    let area1 = (boxes[b1_offset + 2u] - boxes[b1_offset]) * (boxes[b1_offset + 3u] - boxes[b1_offset + 1u]);
    let area2 = (boxes[b2_offset + 2u] - boxes[b2_offset]) * (boxes[b2_offset + 3u] - boxes[b2_offset + 1u]);

    return inter / (area1 + area2 - inter);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let j = global_id.y;

    if (i >= params.num_boxes || j >= params.num_boxes) {
        return;
    }

    if (i >= j) {
        // Only compute upper triangle
        return;
    }

    let iou = compute_iou(i, j);
    if (iou > params.threshold) {
        iou_matrix[i * params.num_boxes + j] = 1u;
    } else {
        iou_matrix[i * params.num_boxes + j] = 0u;
    }
}
