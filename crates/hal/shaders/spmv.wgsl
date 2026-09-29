// Sparse Matrix-Vector Multiply (SpMV) in CSR format
// Computes y = A * x where A is sparse (CSR) and x, y are dense vectors

// The CSR arrays are bound directly below rather than gathered into a struct.
// A WGSL struct field cannot be a runtime-sized array - "Field 'row_ptr' can't be
// dynamically-sized" - so declaring one made this shader fail validation and
// every GPU SpMV a hard error. The struct was also never used: nothing
// constructed or passed one.

// Four storage buffers, which is the device limit; this needed five.
//
// The three CSR arrays are concatenated into one, which is natural since CSR is
// already flat: `[0, n_rows+1)` is row_ptr, then nnz column indices, then nnz
// values. Their lengths come from the uniform rather than from separate bindings.
struct Params {
    row_ptr_len: u32,
    col_offset: u32,
    val_offset: u32,
    vec_len: u32,
}

@group(0) @binding(0) var<storage, read> csr: array<u32>;      // row_ptr then col_indices
@group(0) @binding(1) var<storage, read> values: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn spmv_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    
    // Bounds check
    if (row >= params.row_ptr_len - 1u) {
        return;
    }
    
    let row_start = csr[row];
    let row_end = csr[row + 1u];
    
    var sum: f32 = 0.0;
    for (var i = row_start; i < row_end; i = i + 1u) {
        let col = csr[params.col_offset + i];
        let val = values[i];
        sum = sum + val * x[col];
    }
    
    y[row] = sum;
}
