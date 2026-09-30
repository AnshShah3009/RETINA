// READ-ONLY DIAGNOSTIC. Does not modify the repo's source; run from crates/hal.
// Dumps the real GPU convolve_2d output so the wrong border mapping can be identified.
use cv_core::{CpuTensor, Storage, Tensor, TensorShape};
use cv_hal::context::{BorderMode, ComputeContext};
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
use cv_hal::{CpuBackend, GpuContext};

fn values(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            lo + ((s >> 33) as f32 / u32::MAX as f32) * (hi - lo)
        })
        .collect()
}

#[test]
fn dump() {
    run();
}

fn run() {
    let h = 33usize;
    let w = 37usize;
    let g1 = [0.0625f32, 0.25, 0.375, 0.25, 0.0625];
    let mut kv = Vec::with_capacity(25);
    for r in g1 {
        for c in g1 {
            kv.push(r * c);
        }
    }
    let kernel: CpuTensor<f32> = Tensor::from_vec(kv, TensorShape::new(1, 5, 5)).unwrap();
    let input: CpuTensor<f32> =
        Tensor::from_vec(values(h * w, 41, 0.0, 255.0), TensorShape::new(1, h, w)).unwrap();
    let src = input.storage.as_slice().unwrap().to_vec();

    let cpu = CpuBackend::new().unwrap();
    let g = futures::executor::block_on(GpuContext::init_global()).expect("gpu");
    let gi = input.to_gpu_ctx(&g).unwrap();
    let gk = kernel.to_gpu_ctx(&g).unwrap();

    // Which input index does the GPU actually read for each offset at (0,0)?
    // Invert the convolution: for an impulse-ish probe we instead print the raw
    // GPU output row/column so the mapping can be fitted offline.
    for (name, mode) in [
        ("reflect", BorderMode::Reflect),
        ("reflect101", BorderMode::Reflect101),
        ("wrap", BorderMode::Wrap),
    ] {
        let c = cpu.convolve_2d(&input, &kernel, mode).unwrap();
        let go = g.convolve_2d(&gi, &gk, mode).unwrap();
        let back = go.to_cpu().unwrap();
        let cs = c.storage.as_slice().unwrap();
        let gs = back.storage.as_slice().unwrap();
        println!(
            "### mode={name} ndiff={} ",
            cs.iter()
                .zip(gs)
                .filter(|(a, b)| (*a - *b).abs() > 0.01)
                .count()
        );
        // Print a 9x9 patch of the top-left corner, CPU then GPU, and delta.
        for y in 0..9 {
            let mut cl = String::new();
            let mut gl = String::new();
            let mut dl = String::new();
            for x in 0..9 {
                let i = y * w + x;
                cl.push_str(&format!("{:9.3} ", cs[i]));
                gl.push_str(&format!("{:9.3} ", gs[i]));
                dl.push_str(&format!("{:9.3} ", gs[i] - cs[i]));
            }
            println!("  c y={y} {cl}");
            println!("  g y={y} {gl}");
            println!("  d y={y} {dl}");
        }
        // where do diffs live?
        let mut rows = std::collections::BTreeSet::new();
        let mut cols = std::collections::BTreeSet::new();
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if (cs[i] - gs[i]).abs() > 0.01 {
                    rows.insert(y);
                    cols.insert(x);
                }
            }
        }
        let rows: Vec<usize> = rows.into_iter().collect();
        let cols: Vec<usize> = cols.into_iter().collect();
        println!("  rows differing ({rows:?})");
        println!("  cols differing ({cols:?})");
        println!();
    }

    // Print the input top-left 10x10 so offline mapping work is checkable.
    println!("### input top-left 10x10");
    for y in 0..10 {
        let mut s = String::new();
        for x in 0..10 {
            s.push_str(&format!("{:9.3} ", src[y * w + x]));
        }
        println!("  y={y} {s}");
    }
}
