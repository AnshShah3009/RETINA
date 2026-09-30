// READ-ONLY DIAGNOSTIC #5. Exact readout of the index pair the GPU fetches, per pixel.
//
// Delta kernel (single tap at [0][0]) and in[i] = i, so out[x][y] IS the flat index
// the GPU fetched. Decompose it and compare with the correct (map_x(x-2), map_y(y-2)).
use cv_core::{CpuTensor, Storage, Tensor, TensorShape};
use cv_hal::context::{BorderMode, ComputeContext};
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
use cv_hal::{CpuBackend, GpuContext};

const H: usize = 33;
const W: usize = 37;

fn reflect(c: i64, n: i64) -> i64 {
    if n == 1 {
        return 0;
    }
    let p = 2 * n;
    let mut x = c % p;
    if x < 0 {
        x += p;
    }
    if x >= n {
        x = p - x - 1;
    }
    x
}

#[test]
fn readout2() {
    let g = futures::executor::block_on(GpuContext::init_global()).expect("gpu");
    let cpu = CpuBackend::new().unwrap();
    let mut kv = vec![0.0f32; 25];
    kv[0] = 1.0;
    let kernel: CpuTensor<f32> = Tensor::from_vec(kv, TensorShape::new(1, 5, 5)).unwrap();
    let vals: Vec<f32> = (0..H * W).map(|i| i as f32).collect();
    let input: CpuTensor<f32> = Tensor::from_vec(vals, TensorShape::new(1, H, W)).unwrap();
    let gi = input.to_gpu_ctx(&g).unwrap();
    let gk = kernel.to_gpu_ctx(&g).unwrap();

    for (name, mode) in [
        ("reflect", BorderMode::Reflect),
        ("reflect101", BorderMode::Reflect101),
        ("wrap", BorderMode::Wrap),
    ] {
        let go = g.convolve_2d(&gi, &gk, mode).unwrap();
        let back = go.to_cpu().unwrap();
        let gs = back.storage.as_slice().unwrap();
        let ccpu = cpu.convolve_2d(&input, &kernel, mode).unwrap();
        let cs = ccpu.storage.as_slice().unwrap();

        println!("### {name}");
        // For each output pixel, the coords the kernel would request are
        // cx = x-2, cy = y-2 (only tap at [0][0]).
        println!("  (out_x, out_y) -> (cpu_ix, cpu_iy) | (gpu_ix, gpu_iy)");
        let mut nbad = 0;
        for y in 0..H.min(6) {
            for x in 0..W.min(6) {
                let i = y * W + x;
                let cf = cs[i] as i64;
                let gf = gs[i] as i64;
                let (cix, ciy) = (cf % W as i64, cf / W as i64);
                let (gix, giy) = (gf % W as i64, gf / W as i64);
                let exp_ix = reflect(x as i64 - 2, W as i64);
                let exp_iy = reflect(y as i64 - 2, H as i64);
                let okc = cix == exp_ix && ciy == exp_iy;
                let okg = gix == exp_ix && giy == exp_iy;
                if !okg {
                    nbad += 1;
                }
                println!(
                    "  ({x:2},{y:2}) exp=({exp_ix:2},{exp_iy:2}) cpu=({cix:2},{ciy:2}){} gpu=({gix:2},{giy:2}){}",
                    if okc { "" } else { "  <-- CPU OFF" },
                    if okg { "" } else { "  <-- GPU WRONG" }
                );
            }
        }
        println!("  wrong in first 6x6: {nbad}");
        // Also the far corner
        for (x, y) in [(36usize, 0usize), (0, 32), (36, 32)] {
            let i = y * W + x;
            let cf = cs[i] as i64;
            let gf = gs[i] as i64;
            let (cix, ciy) = (cf % W as i64, cf / W as i64);
            let (gix, giy) = (gf % W as i64, gf / W as i64);
            let (eix, eiy) = (
                reflect(x as i64 - 2, W as i64),
                reflect(y as i64 - 2, H as i64),
            );
            println!(
                "  ({x:2},{y:2}) exp=({eix:2},{eiy:2}) cpu=({cix:2},{ciy:2}) gpu=({gix:2},{giy:2})"
            );
        }
        println!();
    }
}
