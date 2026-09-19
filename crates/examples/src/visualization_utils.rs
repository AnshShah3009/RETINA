use cv_core::{KeyPoints, Matches};
use image::{GrayImage, Rgb, RgbImage};
use std::cmp::max;

/// Draw matches between two images side-by-side.
///
/// Returns an RGB image with lines connecting matched keypoints.
/// Matches with mask value 1 (or no mask) are Green (Inliers).
/// Matches with mask value 0 are Red (Outliers).
pub fn draw_matches(
    img1: &GrayImage,
    kps1: &KeyPoints,
    img2: &GrayImage,
    kps2: &KeyPoints,
    matches: &Matches,
) -> RgbImage {
    let (w1, h1) = img1.dimensions();
    let (w2, h2) = img2.dimensions();

    let total_width = w1 + w2;
    let total_height = max(h1, h2);

    let mut output = RgbImage::new(total_width, total_height);

    // Copy Img1 to left
    for y in 0..h1 {
        for x in 0..w1 {
            let p = img1.get_pixel(x, y)[0];
            output.put_pixel(x, y, Rgb([p, p, p]));
        }
    }

    // Copy Img2 to right
    for y in 0..h2 {
        for x in 0..w2 {
            let p = img2.get_pixel(x, y)[0];
            output.put_pixel(x + w1, y, Rgb([p, p, p]));
        }
    }

    let green = Rgb([0, 255, 0]);
    let red = Rgb([255, 0, 0]);

    for (i, m) in matches.matches.iter().enumerate() {
        let is_inlier = matches
            .mask
            .as_ref()
            .map(|mask| mask[i] > 0)
            .unwrap_or(true);
        let color = if is_inlier { green } else { red };

        let kp1 = kps1.keypoints[m.query_idx as usize];
        let kp2 = kps2.keypoints[m.train_idx as usize];

        let p1 = (kp1.x.round() as i32, kp1.y.round() as i32);
        let p2 = ((kp2.x.round() as i32) + w1 as i32, kp2.y.round() as i32);

        draw_line_segment(&mut output, p1, p2, color);
        draw_circle(&mut output, p1, 2, color); // Tiny circle (radius 2)
        draw_circle(&mut output, p2, 2, color);
    }

    output
}

// Simple Bresenham's line algorithm
fn draw_line_segment(img: &mut RgbImage, p1: (i32, i32), p2: (i32, i32), color: Rgb<u8>) {
    let (mut x0, mut y0) = p1;
    let (x1, y1) = p2;

    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        if x0 >= 0 && x0 < img.width() as i32 && y0 >= 0 && y0 < img.height() as i32 {
            img.put_pixel(x0 as u32, y0 as u32, color);
        }

        if x0 == x1 && y0 == y1 {
            break;
        }

        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

// Simple circle drawing
fn draw_circle(img: &mut RgbImage, center: (i32, i32), radius: i32, color: Rgb<u8>) {
    let (cx, cy) = center;
    let r2 = radius * radius;

    for y in (cy - radius)..=(cy + radius) {
        for x in (cx - radius)..=(cx + radius) {
            if (x - cx).pow(2) + (y - cy).pow(2) <= r2
                && x >= 0
                && x < img.width() as i32
                && y >= 0
                && y < img.height() as i32
            {
                img.put_pixel(x as u32, y as u32, color);
            }
        }
    }
}
