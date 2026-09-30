// READ-ONLY DIAGNOSTIC #7. Read out the FULL border map for coords -R..+R using a
// (2R+1)x(2R+1) delta kernel, and print it next to the correct reflect map.
use cv_core::{CpuTensor, Storage, Tensor, TensorShape};
use cv_hal::context::{BorderMode, ComputeContext};
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
use cv_hal::{CpuBackend, GpuContext};

const R: usize = 8;
const KS: usize = 2 * R + 1;

fn probe(w: usize, h: usize, mode: BorderMode<f32>, label: &str) {
    let g = futures::executor::block_on(GpuContext::init_global()).expect("gpu");
    let _cpu = CpuBackend::new().unwrap();
    let mut kv = vec![0.0f32; KS * KS];
    kv[0] = 1.0;
    let kernel: CpuTensor<f32> = Tensor::from_vec(kv, TensorShape::new(1, KS, KS)).unwrap();
    let vals: Vec<f32> = (0..h * w).map(|i| i as f32).collect();
    let input: CpuTensor<f32> = Tensor::from_vec(vals, TensorShape::new(1, h, w)).unwrap();
    let gi = input.to_gpu_ctx(&g).unwrap();
    let gk = kernel.to_gpu_ctx(&g).unwrap();
    let go = g.convolve_2d(&gi, &gk, mode).unwrap();
    let back = go.to_cpu().unwrap();
    let gs = back.storage.as_slice().unwrap();

    // X map along a mid row
    let ymid = h / 2;
    let mut got = String::new();
    let mut want = String::new();
    for x in 0..(2 * R + 1) {
        if x >= w {
            break;
        }
        let coord = x as i64 - R as i64;
        let flat = gs[ymid * w + x] as i64;
        got.push_str(&format!("{:>4}", flat % w as i64));
        let p = 2 * w as i64;
        let mut c = coord % p;
        if c < 0 {
            c += p;
        }
        if c >= w as i64 {
            c = p - c - 1;
        }
        want.push_str(&format!("{:>4}", c));
    }
    println!("### {label} w={w} h={h}  ({label})");
    println!(
        "  coord :{}",
        (-(R as i64)..=(R as i64))
            .map(|c| format!("{:>4}", c))
            .collect::<String>()
    );
    println!("  gpu ix: {got}");
    println!("  want ix:{want}");
    println!();
}

#[test]
fn full() {
    for (w, h) in [
        (16usize, 8usize),
        (37, 33),
        (9, 40),
        (40, 9),
        (32, 8),
        (8, 8),
    ] {
        probe(w, h, BorderMode::Reflect, "reflect");
    }
}
