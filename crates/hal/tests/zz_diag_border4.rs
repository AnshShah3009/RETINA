// READ-ONLY DIAGNOSTIC #4. Direct readout of the GPU's effective border index map.
//
// Use a 5x5 kernel with a single non-zero tap at [0][0] (value 1). Then
//     out(x,y) = in[ map_x(x-2) + w * map_y(y-2) ]
// because cy=cx=2 and only kx=ky=0 is non-zero. With in[i] = i, the output value
// *is* the flat index the GPU fetched, so the map can be read off directly:
//     map_x(c) = out mod w      map_y(c) = out div w     for c = -2 .. n-3.
use cv_core::{CpuTensor, Storage, Tensor, TensorShape};
use cv_hal::context::{BorderMode, ComputeContext};
use cv_hal::tensor_ext::{TensorToCpu, TensorToGpu};
use cv_hal::{CpuBackend, GpuContext};

const H: usize = 33;
const W: usize = 37;

#[test]
fn readout() {
    let g = futures::executor::block_on(GpuContext::init_global()).expect("gpu");
    let cpu = CpuBackend::new().unwrap();

    // delta kernel: single tap at (ky=0, kx=0)
    let mut kv = vec![0.0f32; 25];
    kv[0] = 1.0;
    let kernel: CpuTensor<f32> = Tensor::from_vec(kv, TensorShape::new(1, 5, 5)).unwrap();

    // in[i] = i  -> the output value is the flat index the GPU read
    let vals: Vec<f32> = (0..H * W).map(|i| i as f32).collect();
    let input: CpuTensor<f32> = Tensor::from_vec(vals, TensorShape::new(1, H, W)).unwrap();
    let gi = input.to_gpu_ctx(&g).unwrap();
    let gk = kernel.to_gpu_ctx(&g).unwrap();

    for (name, mode) in [
        ("reflect", BorderMode::Reflect),
        ("reflect101", BorderMode::Reflect101),
        ("wrap", BorderMode::Wrap),
        ("replicate", BorderMode::Replicate),
    ] {
        let go = g.convolve_2d(&gi, &gk, mode).unwrap();
        let back = go.to_cpu().unwrap();
        let gs = back.storage.as_slice().unwrap();
        let ccpu = cpu.convolve_2d(&input, &kernel, mode).unwrap();
        let cs = ccpu.storage.as_slice().unwrap();

        // Recover the maps. Use y = H-1 (interior) to read the X map, and x = 0 for Y.
        let mut mapx = vec![-1i64; W + 2];
        let mut mapy = vec![-1i64; H + 2];
        let mut mapx_cpu = vec![-1i64; W + 2];
        let mut mapy_cpu = vec![-1i64; H + 2];

        let yi = H - 1; // last row: y-2 = 31, interior
        for x in 0..W {
            let c = x as i64 - 2;
            if c >= -2 {
                mapx[(c + 2) as usize] = gs[yi * W + x] as i64 % W as i64;
                mapx_cpu[(c + 2) as usize] = cs[yi * W + x] as i64 % W as i64;
            }
        }
        let xi = W - 1;
        for y in 0..H {
            let c = y as i64 - 2;
            if c >= -2 {
                mapy[(c + 2) as usize] = gs[y * W + xi] as i64 / W as i64;
                mapy_cpu[(c + 2) as usize] = cs[y * W + xi] as i64 / W as i64;
            }
        }

        println!("### {name}  (W={W}, H={H})");
        print!("  X map  gpu :");
        for v in &mapx {
            print!(" {v}");
        }
        println!();
        print!("  X map  cpu :");
        for v in &mapx_cpu {
            print!(" {v}");
        }
        println!();
        print!("  Y map  gpu :");
        for v in &mapy {
            print!(" {v}");
        }
        println!();
        print!("  Y map  cpu :");
        for v in &mapy_cpu {
            print!(" {v}");
        }
        println!();
        let badx: Vec<(i64, i64, i64)> = (0..mapx.len())
            .filter(|i| mapx[*i] != mapx_cpu[*i])
            .map(|i| (i as i64 - 2, mapx_cpu[i], mapx[i]))
            .collect();
        let bady: Vec<(i64, i64, i64)> = (0..mapy.len())
            .filter(|i| mapy[*i] != mapy_cpu[*i])
            .map(|i| (i as i64 - 2, mapy_cpu[i], mapy[i]))
            .collect();
        println!("  X diffs (coord, cpu, gpu): {badx:?}");
        println!("  Y diffs (coord, cpu, gpu): {bady:?}");
        println!();
    }
}
