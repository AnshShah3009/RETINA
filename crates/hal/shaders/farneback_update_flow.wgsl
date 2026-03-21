// Flow Update Kernel for Farneback Optical Flow
// Reference: OpenCV optflowgf.cpp lines 407-577
// Updates flow field based on polynomial expansion coefficients

struct FlowUpdateParams {
    src_w: u32,
    src_h: u32,
    stride: u32,
    flow_stride: u32,
}

// Estimated polynomial coefficients from first frame
@group(0) @binding(0) var<storage, read> poly1: array<f32>;
// Estimated polynomial coefficients from second frame
@group(0) @binding(1) var<storage, read> poly2: array<f32>;
// Current flow estimate
@group(0) @binding(2) var<storage, read> flow: array<f32>;
// Output updated flow
@group(0) @binding(3) var<storage, read_write> output_flow: array<f32>;
// Number of bands in polynomial expansion
@group(0) @binding(4) var<uniform> params: FlowUpdateParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.src_w || y >= params.src_h) {
        return;
    }
    
    let idx = y * params.stride + x;
    let flow_idx = y * params.flow_stride + x * 2u;
    
    // Get current flow estimate
    var u = flow[flow_idx + 0u];
    var v = flow[flow_idx + 1u];
    
    // Get polynomial coefficients from both frames
    // poly stores: [f, d, e, a, b, c] per pixel
    let f1 = poly1[idx * 6u + 0u];
    let d1 = poly1[idx * 6u + 1u];
    let e1 = poly1[idx * 6u + 2u];
    let a1 = poly1[idx * 6u + 3u];
    let b1 = poly1[idx * 6u + 4u];
    let c1 = poly1[idx * 6u + 5u];
    
    // Build 2x2 system for flow update
    // Based on: 
    //   (a1 + a2)*dx + b1*dy = -d1 + (a2*u + b2*v)
    //   b1*dx + (c1 + c2)*dy = -e1 + (b2*u + c2*v)
    
    // Simplified: build structure tensor and solve
    var g11 = a1;  // Ixx
    var g12 = b1;  // Ixy
    var g22 = c1;  // Iyy
    
    // Get corresponding coefficients from warped second image location
    let wx = f32(x) + u;
    let wy = f32(y) + v;
    
    // Bilinear sample the polynomial coefficients
    let x0 = i32(floor(wx));
    let y0 = i32(floor(wy));
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let fx = wx - f32(x0);
    let fy = wy - f32(y0);
    
    // Clamp to valid range
    let sx0 = clamp(x0, 0, i32(params.src_w) - 1);
    let sy0 = clamp(y0, 0, i32(params.src_h) - 1);
    let sx1 = clamp(x1, 0, i32(params.src_w) - 1);
    let sy1 = clamp(y1, 0, i32(params.src_h) - 1);
    
    // Sample from poly2 (warped location)
    let idx00 = u32(sy0) * params.stride + u32(sx0);
    let idx10 = u32(sy0) * params.stride + u32(sx1);
    let idx01 = u32(sy1) * params.stride + u32(sx0);
    let idx11 = u32(sy1) * params.stride + u32(sx1);
    
    // Bilinear interpolation of a2, b2, c2
    let a2 = mix(mix(poly2[idx00 * 6u + 3u], poly2[idx10 * 6u + 3u], fx),
                 mix(poly2[idx01 * 6u + 3u], poly2[idx11 * 6u + 3u], fx), fy);
    let b2 = mix(mix(poly2[idx00 * 6u + 4u], poly2[idx10 * 6u + 4u], fx),
                 mix(poly2[idx01 * 6u + 4u], poly2[idx11 * 6u + 4u], fx), fy);
    let c2 = mix(mix(poly2[idx00 * 6u + 5u], poly2[idx10 * 6u + 5u], fx),
                 mix(poly2[idx01 * 6u + 5u], poly2[idx11 * 6u + 5u], fx), fy);
    let d2 = mix(mix(poly2[idx00 * 6u + 1u], poly2[idx10 * 6u + 1u], fx),
                 mix(poly2[idx01 * 6u + 1u], poly2[idx11 * 6u + 1u], fx), fy);
    let e2 = mix(mix(poly2[idx00 * 6u + 2u], poly2[idx10 * 6u + 2u], fx),
                 mix(poly2[idx01 * 6u + 2u], poly2[idx11 * 6u + 2u], fx), fy);
    
    // Structure tensor: G = [[g11+g22, g12], [g12, g22+c2]]
    let g11_new = g11 + a2;
    let g12_new = g12 + b2;
    let g22_new = g22 + c2;
    
    // Right-hand side
    let b1_new = -d1 + (a2 * u + b2 * v + d2);
    let b2_new = -e1 + (b2 * u + c2 * v + e2);
    
    // Solve 2x2 system using Cramer's rule
    let det = g11_new * g22_new - g12_new * g12_new;
    
    if (abs(det) > 0.0001) {
        // Update flow
        u += (g22_new * b1_new - g12_new * b2_new) / det;
        v += (g11_new * b2_new - g12_new * b1_new) / det;
    }
    
    // Store updated flow
    output_flow[flow_idx + 0u] = u;
    output_flow[flow_idx + 1u] = v;
}
