// Vector operations for GPU sparse solver
// All operations work on dense f32 vectors stored in GPU buffers

// === AXPY: y = alpha * x + y ===
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> alpha: f32;

@compute @workgroup_size(256)
fn axpy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&x)) {
        return;
    }
    y[idx] = alpha * x[idx] + y[idx];
}

// === Vector copy: dst = src ===
@group(1) @binding(0) var<storage, read> src: array<f32>;
@group(1) @binding(1) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256)
fn vec_copy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&src)) {
        return;
    }
    dst[idx] = src[idx];
}

// === Dot product (partial reduction per workgroup) ===
@group(2) @binding(0) var<storage, read> a: array<f32>;
@group(2) @binding(1) var<storage, read> b: array<f32>;
@group(2) @binding(2) var<storage, read_write> partial_sums: array<f32>;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn dot_product(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let idx = gid.x;
    let local_idx = lid.x;
    
    // Load and multiply
    if (idx < arrayLength(&a)) {
        shared_data[local_idx] = a[idx] * b[idx];
    } else {
        shared_data[local_idx] = 0.0;
    }
    
    workgroupBarrier();
    
    // Parallel reduction within workgroup
    var stride: u32 = 128u;
    while (stride > 0u) {
        if (local_idx < stride) {
            shared_data[local_idx] = shared_data[local_idx] + shared_data[local_idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    
    // Write partial sum
    if (local_idx == 0u) {
        partial_sums[wid.x] = shared_data[0];
    }
}

// === Scale: x = alpha * x ===
@group(3) @binding(0) var<storage, read_write> v: array<f32>;
@group(3) @binding(1) var<uniform> scale: f32;

@compute @workgroup_size(256)
fn vec_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&v)) {
        return;
    }
    v[idx] = scale * v[idx];
}
