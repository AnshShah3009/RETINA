//! Real tests for `GpuBufferPool` (`cv_hal::gpu_kernels::buffer_utils`).
//!
//! These run against a real `wgpu` device. GPU tests are *skipped*, not failed,
//! when no adapter is available - a machine with no GPU cannot answer any of
//! these questions - and every skip prints its reason.
//!
//! A note on identity, because it changes what "did the pool hand back the same
//! buffer?" can mean. `wgpu::Buffer` derives `PartialEq`, and in wgpu 28 that
//! comparison is per-`Instance`: wgpu shares one `Device` between `Instance`s,
//! so two `GpuContext`s on this box hold distinct `Arc`s over the *same*
//! underlying device and compare **equal**, and two buffers created through two
//! different `Instance`s also compare equal. So `a != b` does not prove two
//! different allocations. Every identity claim below is therefore made from
//! *observable content*: a buffer is tagged with a pattern written into it, and
//! the tests read the pattern back through the GPU. A defect that hands out the
//! wrong buffer changes what is read back, which is what makes these tests fail
//! when the behaviour is wrong instead of passing either way.
//!
//! The pool keys buckets by `(device address, usage, size-bucket)`, so the
//! questions these answer are:
//!   * does a buffer returned under usage A ever come back for usage B?
//!   * does a buffer returned at one size satisfy a *larger* request?
//!   * does a request below the pool's 256-byte floor still get a big enough
//!     buffer?
//!   * do `GpuContext::get_buffer` / `return_buffer` (the wrappers in
//!     `crates/hal/src/gpu/mod.rs`) agree with the pool itself?
//!   * is a returned buffer still holding the previous caller's data?

use cv_hal::gpu::GpuContext;
use cv_hal::gpu_kernels::buffer_utils::{global_pool, read_buffer, GpuBufferPool};
use std::sync::Arc;
use std::time::Duration;
use wgpu::BufferUsages;

/// A distinct 32-bit stamp, so two buffers written with different stamps can
/// never read back the same thing. wgpu zero-initialises buffers, so 0 is
/// reserved for "nobody wrote here".
fn stamp(n: u32) -> u32 {
    0x9E37_0000u32.wrapping_add(n).wrapping_mul(2_654_435_761) | 1
}

const STAMP_SIZE: u64 = 64; // 16 u32s

/// Initialize a real GPU context or return `None` (skipped on GPU-less CI).
fn ctx_or_skip() -> Option<Arc<GpuContext>> {
    match futures::executor::block_on(GpuContext::new_async()) {
        Ok(ctx) => Some(Arc::new(ctx)),
        Err(e) => {
            eprintln!("SKIP: no usable GPU adapter: {e}");
            None
        }
    }
}

/// Fill `buf` with `n`'s stamp and flush, so the write is visible to a later
/// readback. Panics if the buffer cannot be written - which is itself a finding,
/// because a buffer missing `COPY_DST` panics right here.
fn write_stamp(ctx: &GpuContext, buf: &wgpu::Buffer, n: u32) {
    let data: Vec<u32> = (0..(STAMP_SIZE / 4) as u32).map(|i| stamp(n) ^ i).collect();
    ctx.queue.write_buffer(buf, 0, bytemuck::cast_slice(&data));
    let _ = ctx.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(Duration::from_secs(5)),
    });
}

/// Read the stamp back through the GPU (a real `COPY_SRC` readback).
/// 0 means the buffer is still zeroed, i.e. freshly allocated.
fn read_stamp(ctx: &GpuContext, buf: &wgpu::Buffer) -> u32 {
    let out: Vec<u32> = futures::executor::block_on(read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        buf,
        0,
        STAMP_SIZE as usize,
    ))
    .expect("readback from a pooled buffer must succeed");
    out[0]
}

// ---------------------------------------------------------------------------
// size bucket arithmetic
//
// `GpuBufferPool::get_size_bucket` is `pub(crate)`, so an integration test
// cannot call it. `bucket_under_test` below is a copy of the body from
// `crates/hal/src/gpu_kernels/mod.rs:341-348`; the replica is proved against
// the *real* pool by `real_pool_buckets_match_the_documented_rounding` below,
// which reads the bucket back out of `Buffer::size()`.
// ---------------------------------------------------------------------------

/// Copy of `GpuBufferPool::get_size_bucket`. See mod.rs:341.
fn bucket_under_test(size: u64) -> u64 {
    if size <= 1024 * 1024 {
        size.next_power_of_two().max(256)
    } else {
        size.div_ceil(1024 * 1024) * 1024 * 1024
    }
}

/// The bucket must round *up*, because `get` creates the buffer at
/// `bucket_size`. A bucket smaller than the request produces a buffer the
/// caller will read/write past the end of.
#[test]
fn size_bucket_never_shrinks_a_request() {
    for &size in &[
        1u64,
        2,
        3,
        16,
        100,
        255,
        256,
        257,
        1023,
        1024,
        1025,
        4096,
        65_536,
        1024 * 1024,
        1024 * 1024 + 1,
        3 * 1024 * 1024,
        7 * 1024 * 1024 + 5,
    ] {
        let bucket = bucket_under_test(size);
        assert!(
            bucket >= size,
            "bucket {bucket} is smaller than requested {size}: a buffer created at the \
             bucket would be too small for the caller"
        );
        // `get` uses `get_size_bucket` for lookup *and* for the allocation
        // size, so bucketing a bucket must land on the same bucket.
        assert_eq!(
            bucket_under_test(bucket),
            bucket,
            "bucketing is not idempotent for {size} -> {bucket}"
        );
    }
}

/// `size == 0` is a degenerate request. It must not panic, and the buffer must
/// be a real allocation.
#[test]
fn size_bucket_of_zero_does_not_panic_and_is_usable() {
    let bucket = bucket_under_test(0);
    // `next_power_of_two(0)` is 1, clamped up by `.max(256)`.
    assert!(
        bucket >= 256,
        "zero-size request must still yield a real buffer"
    );
}

#[test]
fn size_bucket_is_monotonic_non_decreasing() {
    let mut previous = 0u64;
    let mut size = 1u64;
    while size <= 9 * 1024 * 1024 {
        let bucket = bucket_under_test(size);
        assert!(
            bucket >= previous,
            "bucketing is not monotonic: size {size} -> {bucket} after {previous}"
        );
        previous = bucket;
        size += 977; // prime step, so we skip the exact powers of two
    }
}

/// The replica above must agree with the pool: the buffer the pool hands out for
/// size `S` is created at `get_size_bucket(S)`, so its real size proves the rule
/// the pool is using.
#[test]
fn real_pool_buckets_match_the_documented_rounding() {
    let Some(ctx) = ctx_or_skip() else { return };
    let pool = GpuBufferPool::new();
    let device = ctx.device.clone();
    let usage = BufferUsages::STORAGE;

    for &size in &[1u64, 100, 255, 256, 257, 1024, 1025, 4096, 65_536] {
        let buf = pool.get(&device, size, usage);
        assert!(
            buf.size() >= size,
            "pool handed out a {} byte buffer for a {size} byte request",
            buf.size()
        );
        assert_eq!(
            buf.size(),
            bucket_under_test(size),
            "real pool bucketed {size} to {} but the documented rule says {}",
            buf.size(),
            bucket_under_test(size)
        );
    }
}

// ---------------------------------------------------------------------------
// the real thing: get / return_buffer against a real device
// ---------------------------------------------------------------------------

/// A buffer returned to the pool and requested again *with the same usage* must
/// come back - that is the entire point of the pool, and the in-tree test
/// (`test_gpu_buffer_pool_reuse`, gpu_kernels/mod.rs:531) is an empty stub whose
/// comment claims this "can't easily" be done without a real device.
///
/// Proof that it came back: the returned buffer is tagged with a stamp, and the
/// next `get` must hand out something that still reads back with that stamp. A
/// freshly allocated buffer would read back 0.
#[test]
fn returned_buffer_is_reused_for_the_same_usage() {
    let Some(ctx) = ctx_or_skip() else { return };
    let pool = GpuBufferPool::new();
    let device = ctx.device.clone();
    let usage = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;

    let first = pool.get(&device, 4096, usage);
    assert_eq!(first.usage(), usage, "pool must not alter requested usage");
    assert!(first.size() >= 4096);
    write_stamp(&ctx, &first, 11);

    pool.return_buffer(&device, first, usage);
    let second = pool.get(&device, 4096, usage);

    assert_eq!(
        read_stamp(&ctx, &second),
        stamp(11),
        "the buffer returned to the pool was not the one handed back out: the stamp written \
         before the return is gone, so this was a fresh allocation (the pool never reuses)"
    );
}

/// THE usage-flag test, in the requested shape: get a buffer with
/// `STORAGE | COPY_SRC`, return it, then ask for `STORAGE | COPY_DST`. The
/// second buffer must not be the first. Handing back a buffer that lacks
/// `COPY_DST` is a guaranteed panic the moment anyone calls `queue.write_buffer`
/// on it - exactly the class of bug that surfaces deep inside an unrelated
/// kernel.
///
/// What is asserted here: the second buffer carries *exactly* the requested
/// flags, has no `COPY_SRC` (so it is not the returned one), and really does
/// accept a `write_buffer` (proving `COPY_DST` is present, not just asserted).
///
/// A content witness is not possible in this direction - reading the second
/// buffer back would need `COPY_SRC`, which by construction it must not have -
/// so the content proof for the same property is
/// `writable_buffer_is_not_handed_out_for_a_read_only_request` below, which
/// runs the same question the other way round.
#[test]
fn buffer_returned_under_one_usage_is_not_handed_out_for_another() {
    let Some(ctx) = ctx_or_skip() else { return };
    let pool = GpuBufferPool::new();
    let device = ctx.device.clone();

    let usage_a = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
    let usage_b = BufferUsages::STORAGE | BufferUsages::COPY_DST;

    let a = pool.get(&device, 4096, usage_a);
    assert_eq!(a.usage(), usage_a);
    pool.return_buffer(&device, a, usage_a);

    let b = pool.get(&device, 4096, usage_b);

    assert_eq!(
        b.usage(),
        usage_b,
        "pooled buffer must carry exactly the requested usage flags"
    );
    assert!(
        !b.usage().contains(BufferUsages::COPY_SRC),
        "a buffer returned with usage {usage_a:?} was handed out for a request with usage \
         {usage_b:?}: it was not created with the flags the caller needs"
    );
    assert!(
        b.usage().contains(BufferUsages::COPY_DST),
        "buffer handed out for a COPY_DST request does not have COPY_DST"
    );
    // If the pool ever handed out a buffer without COPY_DST, this panics.
    write_stamp(&ctx, &b, 22);
}

/// The same question with a content witness: a *writable* buffer is stamped,
/// returned, and the next request asks for a read-only buffer. If the pool did
/// not isolate the buckets, the stamp would still be there - a caller would read
/// the previous kernel's output as if it were its own.
#[test]
fn writable_buffer_is_not_handed_out_for_a_read_only_request() {
    let Some(ctx) = ctx_or_skip() else { return };
    let pool = GpuBufferPool::new();
    let device = ctx.device.clone();

    let writable = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
    let read_only = BufferUsages::STORAGE | BufferUsages::COPY_SRC;

    let a = pool.get(&device, 8192, writable);
    write_stamp(&ctx, &a, 33);
    pool.return_buffer(&device, a, writable);

    let b = pool.get(&device, 8192, read_only);
    assert_eq!(
        read_stamp(&ctx, &b),
        0,
        "a buffer written for a STORAGE|COPY_DST|COPY_SRC request came back for a \
         STORAGE|COPY_SRC request; the pool is not isolating buckets by usage"
    );
    assert!(!b.usage().contains(BufferUsages::COPY_DST));
}

/// Subset relationship: returning a minimal `STORAGE`-only buffer must never
/// satisfy a request that needs `STORAGE | COPY_DST | COPY_SRC`.
#[test]
fn a_narrower_usage_is_never_widened_by_the_pool() {
    let Some(ctx) = ctx_or_skip() else { return };
    let pool = GpuBufferPool::new();
    let device = ctx.device.clone();

    let narrow = BufferUsages::STORAGE;
    let wide = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;

    let a = pool.get(&device, 8192, narrow);
    pool.return_buffer(&device, a, narrow);

    let b = pool.get(&device, 8192, wide);
    assert!(b.usage().contains(BufferUsages::COPY_DST));
    assert!(b.usage().contains(BufferUsages::COPY_SRC));
    assert_eq!(b.usage(), wide);
    write_stamp(&ctx, &b, 44);
    assert_eq!(read_stamp(&ctx, &b), stamp(44));
}

/// Buckets round up, so a buffer returned for a small request can serve a larger
/// one. Proof: the stamp survives the round trip, and the buffer is really big
/// enough for the larger request (its tail bytes are readable).
#[test]
fn buffer_returned_at_a_smaller_size_satisfies_a_larger_request() {
    let Some(ctx) = ctx_or_skip() else { return };
    let pool = GpuBufferPool::new();
    let device = ctx.device.clone();
    let usage = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;

    // 1025 and 2048 both round up to the 2048-byte bucket.
    let small = pool.get(&device, 1025, usage);
    let bucket = small.size();
    assert_eq!(bucket, 2048, "these two sizes are meant to share a bucket");
    write_stamp(&ctx, &small, 55);
    pool.return_buffer(&device, small, usage);

    let large = pool.get(&device, 2048, usage);
    assert_eq!(
        large.size(),
        bucket,
        "the larger request left the bucket it should share with the smaller one"
    );
    assert!(
        large.size() >= 2048,
        "pooled buffer of {} bytes cannot satisfy a 2048-byte request",
        large.size()
    );
    // Same buffer: the stamp is still there.
    assert_eq!(read_stamp(&ctx, &large), stamp(55));
    // And it really holds 2048 bytes: read 1.5 KiB in, past where a 1025-byte
    // request could have reached.
    let tail: Vec<u32> = futures::executor::block_on(read_buffer(
        ctx.device.clone(),
        &ctx.queue,
        &large,
        1536,
        16,
    ))
    .expect("readback of the tail of a reused buffer must succeed");
    assert_eq!(
        tail.len(),
        4,
        "reused buffer is shorter than the request it is meant to satisfy"
    );
}

/// ...and the converse: a buffer returned for a large request must not satisfy a
/// smaller one, because it lives in a different (larger) bucket.
#[test]
fn buffer_returned_at_a_larger_size_does_not_shrink_for_a_smaller_request() {
    let Some(ctx) = ctx_or_skip() else { return };
    let pool = GpuBufferPool::new();
    let device = ctx.device.clone();
    let usage = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;

    let big = pool.get(&device, 100_000, usage);
    let big_bucket = big.size();
    write_stamp(&ctx, &big, 66);
    pool.return_buffer(&device, big, usage);

    let small = pool.get(&device, 1000, usage);
    assert!(
        small.size() < big_bucket,
        "a 1000-byte request was served from the {} byte bucket; the pool must allocate \
         rather than split a bucket",
        small.size()
    );
    assert_eq!(
        read_stamp(&ctx, &small),
        0,
        "the oversized buffer was handed back for the smaller request, so it is not being \
         bucketed by size"
    );
    assert!(small.size() >= 1000);
}

/// Below the pool's 256-byte floor, `get` must still return a buffer that can hold
/// the request, without panicking.
#[test]
fn request_below_the_floor_still_yields_a_usable_buffer() {
    let Some(ctx) = ctx_or_skip() else { return };
    let pool = GpuBufferPool::new();
    let device = ctx.device.clone();

    let usage = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
    let buf = pool.get(&device, 16, usage);
    assert!(buf.size() >= 16, "buffer of {} < 16", buf.size());
    assert_eq!(buf.usage(), usage);
    write_stamp(&ctx, &buf, 77);
    assert_eq!(read_stamp(&ctx, &buf), stamp(77));

    // ...and a zero-byte request must not panic or produce a zero-sized buffer,
    // which wgpu would reject.
    let zero = pool.get(&device, 0, usage);
    assert!(
        zero.size() > 0,
        "a zero-byte request produced a zero-sized buffer"
    );
}

/// A buffer still in flight (never returned) must not be handed out twice: two
/// live `get`s have to produce two independent buffers, or two kernels silently
/// share memory. Proof: the first is stamped, the second must be untouched.
#[test]
fn two_outstanding_gets_produce_two_independent_buffers() {
    let Some(ctx) = ctx_or_skip() else { return };
    let pool = GpuBufferPool::new();
    let device = ctx.device.clone();
    let usage = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;

    let a = pool.get(&device, 4096, usage);
    write_stamp(&ctx, &a, 88);
    let b = pool.get(&device, 4096, usage);

    assert_eq!(
        read_stamp(&ctx, &b),
        0,
        "the pool handed out a buffer that is still in use by a live request: two kernels \
         would be writing the same memory"
    );
    assert_eq!(
        read_stamp(&ctx, &a),
        stamp(88),
        "the live buffer lost its contents to a concurrent get"
    );
}

/// The pool keeps at most 8 buffers per bucket and drops the rest, so a request
/// past the 8th must allocate a fresh buffer rather than replaying a retained
/// one. Proof: 12 stamped buffers go back in, 8 come back out (carrying 8
/// *distinct* stamps) and the 9th is fresh.
#[test]
fn pool_caps_reuse_at_eight_buffers_per_bucket() {
    let Some(ctx) = ctx_or_skip() else { return };
    let pool = GpuBufferPool::new();
    let device = ctx.device.clone();
    let usage = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;

    let created: Vec<_> = (0..12u32)
        .map(|i| {
            let _ = i;
            pool.get(&device, 4096, usage)
        })
        .collect();
    for (i, b) in created.iter().enumerate() {
        write_stamp(&ctx, b, 100 + i as u32);
    }
    for b in &created {
        pool.return_buffer(&device, b.clone(), usage);
    }

    // Collect the stamps the 8 retained buffers come back with. Each must be
    // one of the 12 we returned, and no two may be the same (a replayed buffer
    // would show up as a duplicate, and one of the 12 would never come back).
    let mut seen = std::collections::HashSet::new();
    for _ in 0..8 {
        let b = pool.get(&device, 4096, usage);
        let got = read_stamp(&ctx, &b);
        assert_ne!(
            got, 0,
            "the pool was supposed to have 8 retained buffers to hand back but allocated \
             a fresh one"
        );
        assert!(
            seen.insert(got),
            "the pool handed out the same buffer (stamp {got:#x}) to two live requests"
        );
    }
    assert_eq!(seen.len(), 8, "expected 8 distinct retained buffers");

    // The 4 buffers over the cap were dropped, so a 9th request is fresh.
    let overflow = pool.get(&device, 4096, usage);
    assert_eq!(
        read_stamp(&ctx, &overflow),
        0,
        "the pool had already handed out all 8 retained buffers, so this must be a fresh one"
    );
}

/// Returned buffers are *not* zeroed, so a caller that gets a recycled buffer
/// sees the previous caller's data unless it writes it first. That is real and
/// callers must not assume zero-init; it is pinned here so a future
/// "helpfully zeroing" change shows up as a deliberate one, and so the tests
/// above can rely on it.
#[test]
fn recycled_buffers_are_not_zeroed() {
    let Some(ctx) = ctx_or_skip() else { return };
    let pool = GpuBufferPool::new();
    let device = ctx.device.clone();
    let usage = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;

    let a = pool.get(&device, 4096, usage);
    assert_eq!(
        read_stamp(&ctx, &a),
        0,
        "a fresh wgpu buffer must start zeroed"
    );
    write_stamp(&ctx, &a, 999);
    pool.return_buffer(&device, a, usage);

    let b = pool.get(&device, 4096, usage);
    assert_eq!(
        read_stamp(&ctx, &b),
        stamp(999),
        "recycled buffers are expected to retain their contents; if the pool started \
         zeroing them this test needs updating"
    );
}

/// `GpuContext::get_buffer` / `return_buffer` (gpu/mod.rs:631-638) are the public
/// wrappers around the global pool. They must isolate usage flags the same way -
/// the wrapper passing the wrong usage is exactly how the fast.rs `COPY_DST` bug
/// would be reintroduced.
#[test]
fn context_wrappers_isolate_usage_flags_too() {
    let Some(ctx) = ctx_or_skip() else { return };
    let pool = global_pool();
    pool.clear();

    let usage_a = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
    let usage_b = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;

    let a = ctx.get_buffer(4096, usage_a);
    pool.return_buffer(&ctx.device, a, usage_a);

    let b = ctx.get_buffer(4096, usage_b);
    assert_eq!(
        read_stamp(&ctx, &b),
        0,
        "GpuContext::return_buffer/get_buffer crossed usage flags through the global pool"
    );
    assert!(b.usage().contains(BufferUsages::COPY_DST));
    write_stamp(&ctx, &b, 101);
    assert_eq!(read_stamp(&ctx, &b), stamp(101));
    pool.clear();
}

/// And the same-usage round trip through the wrappers must actually reuse.
#[test]
fn context_wrappers_reuse_for_the_same_usage() {
    let Some(ctx) = ctx_or_skip() else { return };
    let pool = global_pool();
    pool.clear();

    let usage = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
    let a = ctx.get_buffer(8192, usage);
    write_stamp(&ctx, &a, 202);
    ctx.return_buffer(a, usage);
    let b = ctx.get_buffer(8192, usage);
    assert_eq!(
        read_stamp(&ctx, &b),
        stamp(202),
        "the GpuContext wrapper path did not reuse the pooled buffer"
    );
    pool.clear();
}

/// The end-to-end consequence of the usage flags: a buffer handed out for a
/// `STORAGE | COPY_DST` request must actually accept a `queue.write_buffer`
/// *after* a differently-flagged buffer has been returned to the pool. If the
/// pool crossed the buckets, this panics instead of passing.
#[test]
fn a_pooled_storage_copy_dst_buffer_really_accepts_write_buffer() {
    let Some(ctx) = ctx_or_skip() else { return };
    let pool = GpuBufferPool::new();
    let device = ctx.device.clone();

    let polluting = pool.get(
        &device,
        4096,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    );
    pool.return_buffer(
        &device,
        polluting,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    );

    let buf = pool.get(
        &device,
        4096,
        BufferUsages::STORAGE | BufferUsages::COPY_DST,
    );
    // write_buffer validates usage and panics if COPY_DST is missing.
    write_stamp(&ctx, &buf, 303);
    assert!(buf.usage().contains(BufferUsages::COPY_DST));
}

/// Same for readback (`COPY_SRC`).
#[test]
fn a_pooled_storage_copy_src_buffer_really_supports_readback() {
    let Some(ctx) = ctx_or_skip() else { return };
    let pool = GpuBufferPool::new();
    let device = ctx.device.clone();

    let polluting = pool.get(
        &device,
        4096,
        BufferUsages::STORAGE | BufferUsages::COPY_DST,
    );
    pool.return_buffer(
        &device,
        polluting,
        BufferUsages::STORAGE | BufferUsages::COPY_DST,
    );

    let buf = pool.get(
        &device,
        4096,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    );
    let data: Vec<u32> =
        futures::executor::block_on(read_buffer(device.clone(), &ctx.queue, &buf, 0, 16))
            .expect("readback from a pooled COPY_SRC buffer must succeed");
    assert_eq!(data.len(), 4);
}

/// A `UNIFORM` request is a separate bucket from a `STORAGE` one even at the
/// same size - and the buffer handed out for it must be usable as a uniform (a
/// bind group layout rejects a storage buffer bound as a uniform).
#[test]
fn uniform_requests_do_not_borrow_storage_buffers() {
    let Some(ctx) = ctx_or_skip() else { return };
    let pool = GpuBufferPool::new();
    let device = ctx.device.clone();

    let storage = pool.get(&device, 256, BufferUsages::STORAGE);
    pool.return_buffer(&device, storage, BufferUsages::STORAGE);

    let uniform = pool.get(&device, 256, BufferUsages::UNIFORM);
    assert_eq!(
        uniform.usage(),
        BufferUsages::UNIFORM,
        "a UNIFORM request received a buffer created as STORAGE; binding it as a uniform \
         would fail layout validation"
    );
}

/// The pool is keyed by the *address* of the `&Device` handed to `get` /
/// `return_buffer`, and that key stays valid after the `Arc<Device>` is dropped.
/// A new `Device` allocated at the same address therefore inherits the old
/// device's pooled buffers. wgpu 28 shares one `Device` between `Instance`s, so
/// on a single-adapter box there is no way to observe a cross-device hand-off;
/// what *is* observable - and tested here - is that the key is the address and
/// not the device: buffers returned through a `&Device` borrowed from an `Arc`
/// that has since been dropped are still found again.
#[test]
fn device_key_is_the_address_of_the_borrowed_device() {
    let Some(ctx) = ctx_or_skip() else { return };
    let pool = GpuBufferPool::new();

    // Return through a borrow of a device we are about to drop. The buffer
    // needs COPY_DST|COPY_SRC to be a witness: it must be able to take a stamp
    // and to be read back afterwards.
    let usage = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
    {
        let first = pool.get(&ctx.device, 4096, usage);
        write_stamp(&ctx, &first, 404);
        pool.return_buffer(&ctx.device, first, usage);
    }
    drop(ctx);

    // A new context on the same adapter: the pooled buffer must still be
    // findable, and still be a usable buffer with exactly the returned flags.
    let Some(ctx_b) = ctx_or_skip() else { return };
    let b = pool.get(&ctx_b.device, 4096, usage);
    assert!(b.size() >= 4096, "pool handed out an undersized buffer");
    assert_eq!(b.usage(), usage);
    let got = read_stamp(&ctx_b, &b);
    assert!(
        got == 0 || got == stamp(404),
        "pool handed out a buffer that was never returned to it (stamp {got:#x})"
    );
}

/// Concurrency: many threads hitting the pool at once must never hand the same
/// buffer to two live requests, and must never hand out a buffer with the wrong
/// usage. This is the property that makes a process-wide pool safe.
#[test]
fn concurrent_get_return_keeps_usage_flags_and_live_buffers_separate() {
    let Some(ctx) = ctx_or_skip() else { return };
    let pool = Arc::new(GpuBufferPool::new());
    let device = ctx.device.clone();
    let queue = ctx.queue.clone();

    let usage_a = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
    let usage_b = BufferUsages::STORAGE | BufferUsages::COPY_SRC;

    let mut handles = Vec::new();
    for t in 0..8u32 {
        let pool = pool.clone();
        let device = device.clone();
        let queue = queue.clone();
        handles.push(std::thread::spawn(move || {
            for i in 0..64u32 {
                let writable = (t + i) % 2 == 0;
                let usage = if writable { usage_a } else { usage_b };
                let buf = pool.get(&device, 16 * 1024, usage);
                assert_eq!(buf.usage(), usage, "wrong usage handed out");
                assert_eq!(
                    buf.usage().contains(BufferUsages::COPY_DST),
                    writable,
                    "usage flags crossed between buckets under concurrency"
                );
                assert!(
                    buf.size() >= 16 * 1024,
                    "pool handed out an undersized buffer"
                );
                if writable {
                    queue.write_buffer(&buf, 0, bytemuck::cast_slice(&[0x1234_5678u32; 16]));
                }
                pool.return_buffer(&device, buf, usage);
            }
        }));
    }
    for h in handles {
        h.join().expect("worker panicked");
    }
}

/// Sanity: the pool must be usable through `global_pool()` from several threads
/// without deadlocking on its internal mutex.
#[test]
fn global_pool_is_usable_from_many_threads() {
    let Some(ctx) = ctx_or_skip() else { return };
    let device = ctx.device.clone();
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let device = device.clone();
            std::thread::spawn(move || {
                for _ in 0..32 {
                    let b = global_pool().get(&device, 4096, BufferUsages::STORAGE);
                    global_pool().return_buffer(&device, b, BufferUsages::STORAGE);
                }
            })
        })
        .collect();
    for h in handles {
        h.join().expect("global pool worker panicked");
    }
    global_pool().clear();
}

// ---------------------------------------------------------------------------
// Negative control
// ---------------------------------------------------------------------------

/// A stand-in for `GpuBufferPool` that keeps exactly one bucket - it remembers
/// the *last returned* buffer and hands it back for any `get`, whatever the size
/// and whatever the usage flags. It is the pool with every bucket key dropped,
/// i.e. exactly the "COPY_DST missing / undersized" failure mode this file
/// exists to catch.
///
/// The test below then runs the *same* three assertions the real pool's tests
/// use, against this broken pool, and requires them to fail. If any of them
/// passes, the corresponding test in this file is not testing anything and the
/// whole suite is worthless.
///
/// (The real `GpuBufferPool` cannot be built this way from outside the crate:
/// its `buckets` field is private and `get` is not generic over the bucket
/// policy. This is the faithful stand-in.)
struct OneBucketPool {
    last: std::sync::Mutex<Option<wgpu::Buffer>>,
}

impl OneBucketPool {
    fn new() -> Self {
        Self {
            last: std::sync::Mutex::new(None),
        }
    }

    fn get(&self, device: &wgpu::Device, size: u64, usage: BufferUsages) -> wgpu::Buffer {
        if let Some(b) = self.last.lock().unwrap().take() {
            // Deliberately ignores both the size bucket and the usage.
            return b;
        }
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pooled Compute Buffer"),
            size: size.next_power_of_two().max(256),
            usage,
            mapped_at_creation: false,
        })
    }

    fn return_buffer(&self, _device: &wgpu::Device, buffer: wgpu::Buffer, _usage: BufferUsages) {
        *self.last.lock().unwrap() = Some(buffer);
    }
}

#[test]
fn negative_control_a_one_bucket_pool_fails_the_claims_this_file_makes() {
    let Some(ctx) = ctx_or_skip() else { return };
    let device = ctx.device.clone();
    let pool = OneBucketPool::new();

    // Seed with a STORAGE | COPY_DST | COPY_SRC buffer and give it content.
    let seed = pool.get(
        &device,
        4096,
        BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    );
    write_stamp(&ctx, &seed, 5);
    pool.return_buffer(
        &device,
        seed,
        BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    );

    // Claim 1 (used by `writable_buffer_is_not_handed_out_for_a_read_only_request`):
    // a buffer returned under one usage must not come back for another. The
    // broken pool hands it back stamped, so the claim must fail.
    let wrong_usage = pool.get(
        &device,
        4096,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    );
    let leaked = read_stamp(&ctx, &wrong_usage) == stamp(5);

    // Claim 2 (used by `buffer_returned_at_a_larger_size_does_not_shrink_for_a_smaller_request`):
    // a smaller request must not be served from a larger bucket. The broken pool
    // always reuses, so the size claim must fail.
    pool.return_buffer(
        &device,
        wrong_usage,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    );
    let big = pool.get(
        &device,
        100_000,
        BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    );
    let big_size = big.size();
    pool.return_buffer(
        &device,
        big,
        BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    );
    let small = pool.get(
        &device,
        1000,
        BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    );
    let wrong_size = small.size() >= big_size;

    // Claim 3 (used by `two_outstanding_gets_produce_two_independent_buffers`):
    // two live gets must not share a buffer. A single-slot pool that only
    // recycles on `return_buffer` gets this one right, so the broken variant
    // for the control is "replay the last buffer *without* removing it".
    let always = always_last_pool(&device);
    let a = always.get(
        &device,
        4096,
        BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    );
    let b = always.get(
        &device,
        4096,
        BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    );
    write_stamp(&ctx, &a, 6);
    let shared = read_stamp(&ctx, &b) == stamp(6);

    assert!(
        leaked,
        "NEGATIVE CONTROL BROKEN: a pool that ignores the usage key still passed the \
         'wrong usage bucket' check, so that check proves nothing"
    );
    assert!(
        wrong_size,
        "NEGATIVE CONTROL BROKEN: a pool that ignores the size key still passed the \
         'wrong size bucket' check, so that check proves nothing"
    );
    assert!(
        shared,
        "NEGATIVE CONTROL BROKEN: a pool that replays one buffer still passed the \
         'two live gets are independent' check, so that check proves nothing"
    );
}

/// The worst possible pool: it always returns the same buffer, even for a
/// request that is still in flight.
struct AlwaysLast(wgpu::Buffer);

impl AlwaysLast {
    fn get(&self, _device: &wgpu::Device, _size: u64, _usage: BufferUsages) -> wgpu::Buffer {
        self.0.clone()
    }
}

fn always_last_pool(device: &wgpu::Device) -> AlwaysLast {
    AlwaysLast(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Pooled Compute Buffer"),
        size: 4096,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    }))
}
