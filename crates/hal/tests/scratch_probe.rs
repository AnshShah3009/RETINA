#![cfg(feature = "cubecl")]

use cubecl::prelude::*;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};
use cv_hal::gpu_kernels::cubecl_advanced as adv;
use cv_hal::gpu_kernels::cubecl_proto as proto;

type Rt = WgpuRuntime;

fn client() -> ComputeClient<Rt> {
    <Rt as Runtime>::client(&WgpuDevice::DefaultDevice)
}

#[test]
fn normal_orientation_on_a_fronto_parallel_plane() {
    let c = client();
    let ctx = adv::AdvancedContext::<Rt>::new(c.clone());
    let (h, w) = (5usize, 5usize);
    let flat = vec![2.0f32; h * w];
    let t = proto::tensor_from_slice(&c, &flat, vec![h, w]).unwrap();
    let n =
        proto::tensor_to_slice(&c, &adv::depth_to_normals(&ctx, &t, 50.0, 50.0).unwrap()).unwrap();
    let o = (2 * w + 2) * 3;
    println!(
        "interior normal at (2,2) = ({}, {}, {})",
        n[o],
        n[o + 1],
        n[o + 2]
    );
    println!("border   normal at (0,0) = ({}, {}, {})", n[0], n[1], n[2]);
}

#[test]
fn icp_residual_of_a_translated_cloud() {
    let c = client();
    let ctx = adv::AdvancedContext::<Rt>::new(c.clone());
    // 20 source points on a line; target = the same points shifted by 0.5 in x.
    let ns = 20usize;
    let mut source = vec![0.0f32; ns * 3];
    for i in 0..ns {
        source[i * 3] = i as f32;
    }
    let shifted: Vec<f32> = source
        .iter()
        .enumerate()
        .map(|(i, v)| if i % 3 == 0 { v + 0.5 } else { *v })
        .collect();
    let identity = vec![
        1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let tsrc = proto::tensor_from_slice(&c, &source, vec![ns, 3]).unwrap();
    let tsh = proto::tensor_from_slice(&c, &shifted, vec![ns, 3]).unwrap();
    let tti = proto::tensor_from_slice(&c, &identity, vec![4, 4]).unwrap();

    let a =
        proto::tensor_to_slice(&c, &adv::icp_residuals(&ctx, &tsh, &tsrc, &tti).unwrap()).unwrap();
    let b =
        proto::tensor_to_slice(&c, &adv::icp_residuals(&ctx, &tsrc, &tsrc, &tti).unwrap()).unwrap();
    println!("shifted vs original: {:?}", &a[..5]);
    println!("original vs original: {:?}", &b[..5]);
}
