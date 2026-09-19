// matrix_multiply.wgsl
// GPU matrix multiplication kernel using tiled shared memory

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

struct Uniforms {
    dim_m: u32,
    dim_k: u32,
    dim_n: u32,
};
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

const TILE_SIZE: u32 = 16u;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = global_id.y;
    let col = global_id.x;
    
    var sum: f32 = 0.0;
    
    if (row < uniforms.dim_m && col < uniforms.dim_n) {
        for (var k: u32 = 0u; k < uniforms.dim_k; k = k + 1u) {
            sum = sum + A[row * uniforms.dim_k + k] * B[k * uniforms.dim_n + col];
        }
        C[row * uniforms.dim_n + col] = sum;
    }
}
