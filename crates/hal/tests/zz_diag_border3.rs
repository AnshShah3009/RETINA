// READ-ONLY DIAGNOSTIC #3. Dumps full GPU convolve_2d outputs + input to /tmp for
// offline identification of the exact 1-D index mapping the GPU is using.
use cv_core::{CpuTensor, Storage, Tensor, TensorShape};
use cv_hal::context::{BorderMode, ComputeContext};
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
use cv_hal::{CpuBackend, GpuContext};

const H: usize = 33;
const W: usize = 37;

fn values(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
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
fn dump_all() {
    let g = futures::executor::block_on(GpuContext::init_global()).expect("gpu");
    let cpu = CpuBackend::new().unwrap();
    let g1 = [0.0625f32, 0.25, 0.375, 0.25, 0.0625];
    let mut kv = Vec::with_capacity(25);
    for r in g1 {
        for c in g1 {
            kv.push(r * c);
        }
    }
    let kernel: CpuTensor<f32> = Tensor::from_vec(kv, TensorShape::new(1, 5, 5)).unwrap();
    let input: CpuTensor<f32> =
        Tensor::from_vec(values(H * W, 41, 0.0, 255.0), TensorShape::new(1, H, W)).unwrap();
    let src = input.storage.as_slice().unwrap().to_vec();
    let gi = input.to_gpu_ctx(&g).unwrap();
    let gk = kernel.to_gpu_ctx(&g).unwrap();

    let mut out = String::new();
    out.push_str(&format!("{} {}\n", H, W));
    out.push_str("input\n");
    for v in &src {
        out.push_str(&format!("{v:.9}\n"));
    }
    for (name, mode) in [
        ("reflect", BorderMode::Reflect),
        ("reflect101", BorderMode::Reflect101),
        ("wrap", BorderMode::Wrap),
        ("replicate", BorderMode::Replicate),
    ] {
        let c = cpu.convolve_2d(&input, &kernel, mode).unwrap();
        let go = g.convolve_2d(&gi, &gk, mode).unwrap();
        let back = go.to_cpu().unwrap();
        let cs = c.storage.as_slice().unwrap();
        let gs = back.storage.as_slice().unwrap();
        out.push_str(&format!("BEGIN {name} cpu\n"));
        for v in cs {
            out.push_str(&format!("{v:.9}\n"));
        }
        out.push_str(&format!("BEGIN {name} gpu\n"));
        for v in gs {
            out.push_str(&format!("{v:.9}\n"));
        }
    }
    std::fs::write("/tmp/commandcode-1000/-home-Phoenix-RUST/75c9a1af-cf26-4ba5-a448-f495811261e8/scratchpad/gpu_dump.txt", out).unwrap();
    println!("wrote dump");
}
