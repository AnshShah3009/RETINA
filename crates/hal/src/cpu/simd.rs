use wide::f32x8;

/// Convert RGB to Grayscale using SIMD
/// Weights: R: 0.299, G: 0.587, B: 0.114
pub fn rgb_to_gray_simd(rgb: &[u8], gray: &mut [u8]) {
    assert_eq!(rgb.len(), gray.len() * 3);

    // Process 8 pixels at a time (24 bytes of RGB)
    let chunk_size = 8;
    let n = gray.len();
    let n_simd = n - (n % chunk_size);

    let w_r = f32x8::splat(0.299);
    let w_g = f32x8::splat(0.587);
    let w_b = f32x8::splat(0.114);

    for i in (0..n_simd).step_by(chunk_size) {
        let rgb_chunk = &rgb[i * 3..(i + 8) * 3];

        let mut r_arr = [0.0f32; 8];
        let mut g_arr = [0.0f32; 8];
        let mut b_arr = [0.0f32; 8];

        for k in 0..8 {
            r_arr[k] = rgb_chunk[k * 3] as f32;
            g_arr[k] = rgb_chunk[k * 3 + 1] as f32;
            b_arr[k] = rgb_chunk[k * 3 + 2] as f32;
        }

        let r_vec = f32x8::from(r_arr);
        let g_vec = f32x8::from(g_arr);
        let b_vec = f32x8::from(b_arr);

        let gray_vec = r_vec * w_r + g_vec * w_g + b_vec * w_b;
        let gray_arr: [f32; 8] = gray_vec.into();

        for k in 0..8 {
            gray[i + k] = gray_arr[k] as u8;
        }
    }

    // Handle remainder
    for i in n_simd..n {
        let r = rgb[i * 3] as f32;
        let g = rgb[i * 3 + 1] as f32;
        let b = rgb[i * 3 + 2] as f32;
        gray[i] = (0.299 * r + 0.587 * g + 0.114 * b) as u8;
    }
}

/// Compute 1D convolution on a row of pixels using SIMD
///
/// * `src`: Source row data (must be padded or handled by caller for boundaries)
/// * `dst`: Destination row data
/// * `kernel`: Convolution kernel
/// * `radius`: Kernel radius (kernel size = 2 * radius + 1)
#[allow(clippy::needless_range_loop)]
pub fn convolve_row_1d(src: &[f32], dst: &mut [f32], kernel: &[f32], radius: usize) {
    let width = dst.len();
    let k_len = kernel.len();
    assert_eq!(k_len, 2 * radius + 1);

    // Process 8 pixels at a time
    let chunk_size = 8;
    let width_simd = if width >= chunk_size {
        width - chunk_size + 1
    } else {
        0
    };

    for x in (0..width_simd).step_by(chunk_size) {
        let mut sum_v = f32x8::ZERO;

        for k in 0..k_len {
            let w = f32x8::splat(kernel[k]);
            // Load 8 pixels starting from src[x + k]
            let offset = x + k;
            if offset + 8 <= src.len() {
                let mut chunk = [0.0f32; 8];
                chunk.copy_from_slice(&src[offset..offset + 8]);
                sum_v += f32x8::from(chunk) * w;
            }
        }

        let res: [f32; 8] = sum_v.into();
        dst[x..x + 8].copy_from_slice(&res);
    }

    // Scalar fallback for remaining pixels (or entire row when width < 8)
    let scalar_start = if width_simd > 0 {
        width_simd - 1 + chunk_size
    } else {
        0
    };
    for x in scalar_start..width {
        let mut sum = 0.0f32;
        for k in 0..k_len {
            let offset = x + k;
            if offset < src.len() {
                sum += src[offset] * kernel[k];
            }
        }
        dst[x] = sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_to_gray_simd_parity() {
        let rgb = vec![
            255, 0, 0, // Red
            0, 255, 0, // Green
            0, 0, 255, // Blue
            255, 255, 255, // White
            0, 0, 0, // Black
            128, 128, 128, // Gray
            10, 20, 30, 100, 150, 200,
        ];
        let mut gray_simd = vec![0u8; 8];
        let mut gray_scalar = vec![0u8; 8];

        rgb_to_gray_simd(&rgb, &mut gray_simd);

        for i in 0..8 {
            let r = rgb[i * 3] as f32;
            let g = rgb[i * 3 + 1] as f32;
            let b = rgb[i * 3 + 2] as f32;
            gray_scalar[i] = (0.299 * r + 0.587 * g + 0.114 * b) as u8;
        }

        assert_eq!(gray_simd, gray_scalar);
    }

    #[test]
    fn test_convolve_row_1d_simd_parity() {
        let src = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let kernel = vec![0.25, 0.5, 0.25];
        let radius = 1;

        let mut dst_simd = vec![0.0f32; 8];
        let mut dst_scalar = vec![0.0f32; 8];

        // SIMD
        convolve_row_1d(&src, &mut dst_simd, &kernel, radius);

        // Scalar
        for x in 0..8 {
            let mut sum = 0.0;
            for k in 0..3 {
                sum += src[x + k] * kernel[k];
            }
            dst_scalar[x] = sum;
        }

        for i in 0..8 {
            assert!((dst_simd[i] - dst_scalar[i]).abs() < 1e-6);
        }
    }
}
