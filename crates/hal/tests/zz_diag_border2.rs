// READ-ONLY DIAGNOSTIC #2. Probes the GPU's EFFECTIVE border index mapping.
//
// The kernel is separable: K[ky][kx] = g1[ky]*g1[kx], g1 = [0.0625,0.25,0.375,0.25,0.0625],
// so with a column stripe (in[x,y] = 1 iff x == 0) the output depends ONLY on map_x:
//     out(x) = g1[2] * sum_kx g1[kx] * [ map_x(x + kx - 2) == 0 ]
// (the stripe covers every row, so map_y is irrelevant). Same for a row stripe.
use cv_core::{CpuTensor, Storage, Tensor, TensorShape};
use cv_hal::context::{BorderMode, ComputeContext};
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
use cv_hal::{CpuBackend, GpuContext};

const H: usize = 33;
const W: usize = 37;

fn run_case(g: &GpuContext, mode: BorderMode<f32>, kind: &str) -> (Vec<f32>, Vec<f32>) {
    let mut v = vec![0.0f32; H * W];
    match kind {
        "colstripe" => {
            for y in 0..H {
                v[y * W] = 1.0;
            }
        }
        "rowstripe" => {
            for x in 0..W {
                v[x] = 1.0;
            }
        }
        _ => {
            v[0] = 1.0;
        }
    }
    let input: CpuTensor<f32> = Tensor::from_vec(v, TensorShape::new(1, H, W)).unwrap();
    let g1 = [0.0625f32, 0.25, 0.375, 0.25, 0.0625];
    let mut kv = Vec::with_capacity(25);
    for r in g1 {
        for c in g1 {
            kv.push(r * c);
        }
    }
    let kernel: CpuTensor<f32> = Tensor::from_vec(kv, TensorShape::new(1, 5, 5)).unwrap();
    let gi = input.to_gpu_ctx(g).unwrap();
    let gk = kernel.to_gpu_ctx(g).unwrap();
    let go = g.convolve_2d(&gi, &gk, mode).unwrap();
    let back = go.to_cpu().unwrap();
    let gs = back.storage.as_slice().unwrap().to_vec();
    let cpu = CpuBackend::new().unwrap();
    let c = cpu.convolve_2d(&input, &kernel, mode).unwrap();
    (c.storage.as_slice().unwrap().to_vec(), gs)
}

fn dump(name: &str, kind: &str, cpu: &[f32], gpu: &[f32]) {
    println!("@@@ {name} {kind}");
    for y in 0..H {
        let mut s = String::new();
        for x in 0..W {
            let i = y * W + x;
            if (cpu[i] - gpu[i]).abs() > 1e-5 {
                s.push_str(&format!(" ({x}:c{:.6}/g{:.6})", cpu[i], gpu[i]));
            }
        }
        if !s.is_empty() {
            println!("  y={y}{s}");
        }
    }
}

#[test]
fn probe() {
    let g = futures::executor::block_on(GpuContext::init_global()).expect("gpu");
    for (name, mode) in [
        ("reflect", BorderMode::Reflect),
        ("reflect101", BorderMode::Reflect101),
        ("wrap", BorderMode::Wrap),
    ] {
        for kind in ["colstripe", "rowstripe", "impulse"] {
            let (c, s) = run_case(&g, mode, kind);
            dump(name, kind, &c, &s);
        }
    }
}
