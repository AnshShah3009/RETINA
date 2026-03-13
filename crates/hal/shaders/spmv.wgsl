// Sparse Matrix-Vector Multiply (SpMV) in CSR format
// Computes y = A * x where A is sparse (CSR) and x, y are dense vectors

struct SparseMatrix {
    row_ptr: array<u32>,      // size: rows + 1
    col_indices: array<u32>,  // size: nnz
    values: array<f32>,       // size: nnz
}

@group(0) @binding(0) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(1) var<storage, read> col_indices: array<u32>;
@group(0) @binding(2) var<storage, read> values: array<f32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256)
fn spmv_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    
    // Bounds check
    if (row >= arrayLength(&row_ptr) - 1u) {
        return;
    }
    
    let row_start = row_ptr[row];
    let row_end = row_ptr[row + 1u];
    
    var sum: f32 = 0.0;
    for (var i = row_start; i < row_end; i = i + 1u) {
        let col = col_indices[i];
        let val = values[i];
        sum = sum + val * x[col];
    }
    
    y[row] = sum;
}
