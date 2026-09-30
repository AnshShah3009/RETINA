// READ-ONLY DIAGNOSTIC #6. Sweep image sizes and read out the GPU's effective
// border map for coords -3..+3, using a 7x7 delta kernel (cx = 3).
//
// in[i] = i, delta kernel with a single tap at [0][0] => out[x][y] = the flat index
// the GPU fetched. Decomposing that into (ix, iy) gives the true pixel it read.
use cv_core::{CpuTensor, Storage, Tensor, TensorShape};
use cv_hal::context::{BorderMode, ComputeContext};
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
use cv_hal::{CpuBackend, GpuContext};

fn probe(w: usize, h: usize, mode: BorderMode<f32>) {
    let g = futures::executor::block_on(GpuContext::init_global()).expect("gpu");
    let cpu = CpuBackend::new().unwrap();
    let ks = 7usize;
    let mut kv = vec![0.0f32; ks * ks];
    kv[0] = 1.0; // tap at [ky=0][kx=0]
    let kernel: CpuTensor<f32> = Tensor::from_vec(kv, TensorShape::new(1, ks, ks)).unwrap();
    let vals: Vec<f32> = (0..h * w).map(|i| i as f32).collect();
    let input: CpuTensor<f32> = Tensor::from_vec(vals, TensorShape::new(1, h, w)).unwrap();
    let gi = input.to_gpu_ctx(&g).unwrap();
    let gk = kernel.to_gpu_ctx(&g).unwrap();
    let go = g.convolve_2d(&gi, &gk, mode).unwrap();
    let back = go.to_cpu().unwrap();
    let gs = back.storage.as_slice().unwrap();
    let _ = cpu.convolve_2d(&input, &kernel, mode).unwrap();

    let cx = (ks / 2) as i64;
    // X map: read along a row deep inside the image (iy correct/interior)
    let ymid = h / 2;
    let mut xmap = String::new();
    for x in 0..7usize {
        let coord = x as i64 - cx;
        let flat = gs[ymid * w + x] as i64;
        xmap.push_str(&format!(
            "  {}->{}({})",
            coord,
            flat % w as i64,
            flat / w as i64
        ));
    }
    // Y map: read along a column deep inside
    let xmid = w / 2;
    let mut ymap = String::new();
    for y in 0..7usize {
        let coord = y as i64 - cx;
        let flat = gs[y * w + xmid] as i64;
        ymap.push_str(&format!(
            "  {}->{}({})",
            coord,
            flat % w as i64,
            flat / w as i64
        ));
    }
    println!("  w={w:<3} h={h:<3} X:{xmap}");
    println!("  {w:>9} {h:>3} Y:{ymap}");
}

#[test]
fn sweep() {
    println!("### reflect (coord -> ix(iy) read)");
    for (w, h) in [
        (16usize, 8usize),
        (32, 8),
        (16, 32),
        (37, 33),
        (40, 9),
        (9, 40),
        (64, 5),
    ] {
        probe(w, h, BorderMode::Reflect);
    }
    println!("### reflect101");
    for (w, h) in [(16usize, 8usize), (37, 33)] {
        probe(w, h, BorderMode::Reflect101);
    }
    println!("### wrap");
    for (w, h) in [(16usize, 8usize), (37, 33)] {
        probe(w, h, BorderMode::Wrap);
    }
}
