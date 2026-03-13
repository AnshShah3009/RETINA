// Image subtraction kernel: output = a - b
// Works on flat arrays of f32

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_data: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&a) || idx >= arrayLength(&b)) {
        return;
    }
    output_data[idx] = a[idx] - b[idx];
}
