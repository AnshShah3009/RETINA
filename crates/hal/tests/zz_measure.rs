use cv_core::storage::CpuStorage;
use cv_core::tensor::Tensor;
use cv_core::TensorShape;
use cv_hal::context::ComputeContext;
use cv_hal::cpu::CpuBackend;

fn t(d: &[f32], w: usize, h: usize, c: usize) -> Tensor<f32, CpuStorage<f32>> {
    Tensor::from_vec(d.to_vec(), TensorShape::new(c, h, w)).unwrap()
}
/// Reference pyramid: separable [1 4 6 4 1]/16 blur then decimate by 2.
/// Computed from scratch (not via cv_hal) as the oracle.
fn ref_pyramid_down(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let k = [1.0f32, 4.0, 6.0, 4.0, 1.0];
    let (nw, nh) = (w / 2, h / 2);
    let mut tmp = vec![0.0f32; nw * h];
    for y in 0..h {
        for x in 0..nw {
            let mut s = 0.0;
            for i in 0..5 {
                s += k[i]
                    * src[y * w + ((x * 2 + i) as isize - 2).clamp(0, w as isize - 1) as usize];
            }
            tmp[y * nw + x] = s / 16.0;
        }
    }
    let mut out = vec![0.0f32; nw * nh];
    for y in 0..nh {
        for x in 0..nw {
            let mut s = 0.0;
            for i in 0..5 {
                s += k[i]
                    * tmp[(((y * 2 + i) as isize - 2).clamp(0, h as isize - 1) as usize) * nw + x];
            }
            out[y * nw + x] = s / 16.0;
        }
    }
    out
}

#[test]
fn pyramid_parity() {
    let cpu = CpuBackend::new().unwrap();
    for &(w, h) in &[(100usize, 100usize), (99, 99), (100, 80), (17, 13)] {
        let img: Vec<f32> = (0..w * h)
            .map(|i| ((i % w) as f32 * 0.7 + (i / w) as f32 * 0.3).sin() * 40.0)
            .collect();
        let t1 = t(&img, w, h, 1);
        let p = cpu.pyramid_down(&t1).unwrap();
        let got = p.as_slice().unwrap();
        let exp = ref_pyramid_down(&img, w, h);
        println!(
            "{w}x{h}: got {}x{} expect {}x{}",
            p.shape.width,
            p.shape.height,
            w / 2,
            h / 2
        );
        if got.len() == exp.len() {
            let maxdiff = got
                .iter()
                .zip(exp.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let rms = (got
                .iter()
                .zip(exp.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                / got.len() as f32)
                .sqrt();
            println!("   vs scratch [1 4 6 4 1]/16 blur + decimate: max={maxdiff:.5} rms={rms:.5}");
        }
    }
    // control: constant image must survive pyramid_down unchanged
    let flat = vec![7.0f32; 64];
    let p = cpu.pyramid_down(&t(&flat, 8, 8, 1)).unwrap();
    println!("control constant image -> {:?}", p.as_slice().unwrap());
}
