//! Byte-exact contracts of `buffer_utils::create_buffer` and `read_buffer`.
//!
//! Every GPU kernel uploads and reads back through these two helpers, so the
//! copies they make are on every hot path. These tests pin what the bytes on
//! the device and the values handed back must be, independently of how many
//! host copies the helpers take to get there.
//!
//! Skips when no GPU adapter is available.

use cv_hal::gpu::GpuContext;
use cv_hal::gpu_kernels::buffer_utils::{create_buffer, read_buffer};
use wgpu::BufferUsages;

fn ctx_or_skip() -> Option<GpuContext> {
    match GpuContext::new() {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("skipping: no GPU ({e})");
            None
        }
    }
}

fn read<T: bytemuck::Pod + std::fmt::Debug>(
    ctx: &GpuContext,
    buf: &wgpu::Buffer,
    offset: u64,
    size: usize,
) -> Vec<T> {
    pollster::block_on(read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        buf,
        offset,
        size,
    ))
    .expect("readback")
}

#[test]
fn a_storage_upload_reads_back_bit_exact() {
    let Some(ctx) = ctx_or_skip() else { return };
    // Values whose bit patterns would expose a byte-order or offset slip.
    let data: Vec<f32> = (0..1031)
        .map(|i| f32::from_bits(0x3f80_0000 ^ (i as u32).wrapping_mul(2_654_435_761)))
        .collect();
    let buf = create_buffer(
        &ctx.device,
        &data,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    );
    assert_eq!(buf.size(), (data.len() * 4) as u64);
    let got: Vec<f32> = read(&ctx, &buf, 0, data.len() * 4);
    let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
    let want_bits: Vec<u32> = data.iter().map(|v| v.to_bits()).collect();
    assert_eq!(got_bits, want_bits);
}

#[test]
fn a_uniform_upload_is_padded_to_16_bytes_with_zeros() {
    let Some(ctx) = ctx_or_skip() else { return };
    let params = [0xdead_beefu32, 1, 2, 3, 4, 5, 6];
    let buf = create_buffer(
        &ctx.device,
        &params,
        BufferUsages::UNIFORM | BufferUsages::COPY_SRC,
    );
    assert_eq!(buf.size(), 32, "28 bytes of uniform pad up to 32");
    let got: Vec<u32> = read(&ctx, &buf, 0, 32);
    assert_eq!(got, vec![0xdead_beef, 1, 2, 3, 4, 5, 6, 0]);
}

#[test]
fn a_uniform_already_a_multiple_of_16_is_not_padded() {
    let Some(ctx) = ctx_or_skip() else { return };
    let params = [9u32, 8, 7, 6];
    let buf = create_buffer(
        &ctx.device,
        &params,
        BufferUsages::UNIFORM | BufferUsages::COPY_SRC,
    );
    assert_eq!(buf.size(), 16);
    let got: Vec<u32> = read(&ctx, &buf, 0, 16);
    assert_eq!(got, params);
}

#[test]
fn readback_honours_offset_and_truncates_to_whole_elements() {
    let Some(ctx) = ctx_or_skip() else { return };
    let data: Vec<u32> = (100..164).collect();
    let buf = create_buffer(
        &ctx.device,
        &data,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    );
    // 16 bytes in, 21 bytes long: five whole u32s, the sixth is partial and
    // must be dropped rather than read past.
    let got: Vec<u32> = read(&ctx, &buf, 16, 21);
    assert_eq!(got, vec![104, 105, 106, 107, 108]);
}

#[test]
fn readback_of_a_wider_element_type_reinterprets_the_bytes() {
    let Some(ctx) = ctx_or_skip() else { return };
    let data: Vec<u32> = vec![1, 0, 2, 0, 0xffff_ffff, 0x7fff_ffff];
    let buf = create_buffer(
        &ctx.device,
        &data,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    );
    let got: Vec<u64> = read(&ctx, &buf, 0, 24);
    assert_eq!(got, vec![1, 2, 0x7fff_ffff_ffff_ffff]);
}
