use cv_core::Float;

pub(crate) fn get_val_cpu<T: Float>(src: &[T], w: usize, h: usize, x: i32, y: i32) -> T {
    let cx = x.clamp(0, w as i32 - 1) as usize;
    let cy = y.clamp(0, h as i32 - 1) as usize;
    src[cy * w + cx]
}

pub(crate) fn get_pixel_cpu(src: &[f32], w: usize, h: usize, x: i32, y: i32) -> f32 {
    let cx = (x as i32).clamp(0, w as i32 - 1) as usize;
    let cy = (y as i32).clamp(0, h as i32 - 1) as usize;
    src[cy * w + cx]
}

#[allow(clippy::needless_range_loop)]
pub(crate) fn has_9_contiguous_generic<T: Float>(vals: &[T; 16], high: T, low: T) -> bool {
    let mut b_mask = 0u32;
    let mut d_mask = 0u32;
    for i in 0..16 {
        if vals[i] > high {
            b_mask |= 1 << i;
        }
        if vals[i] < low {
            d_mask |= 1 << i;
        }
    }

    let b_mask_ext = b_mask | (b_mask << 16);
    let d_mask_ext = d_mask | (d_mask << 16);

    for i in 0..16 {
        if (b_mask_ext >> i) & 0x1FF == 0x1FF {
            return true;
        }
        if (d_mask_ext >> i) & 0x1FF == 0x1FF {
            return true;
        }
    }
    false
}

pub(crate) fn hamming_dist(a: &[u8], b: &[u8]) -> u32 {
    let mut dist = 0;
    for i in 0..a.len() {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}
