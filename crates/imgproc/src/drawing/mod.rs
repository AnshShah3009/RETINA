//! Drawing primitives for image annotation (OpenCV-equivalent).
//!
//! All functions operate on `image::RgbImage` or `image::GrayImage`.
//! Thickness of -1 means filled (OpenCV convention).
//!
//! # Example
//! ```ignore
//! use cv_imgproc::drawing::*;
//! use image::RgbImage;
//!
//! let mut img = RgbImage::new(640, 480);
//! draw_line(&mut img, (10, 10), (200, 150), RED, 2);
//! draw_circle(&mut img, (320, 240), 50, GREEN, -1);
//! draw_rectangle(&mut img, (50, 50), (200, 200), BLUE, 1);
//! draw_text(&mut img, "Hello RETINA", (10, 400), 2, WHITE);
//! ```

mod drawing_font;

use image::{GrayImage, Rgb, RgbImage};

// ── Color constants ─────────────────────────────────────────────────────────

pub type Color = [u8; 3];

pub const RED: Color = [255, 0, 0];
pub const GREEN: Color = [0, 255, 0];
pub const BLUE: Color = [0, 0, 255];
pub const WHITE: Color = [255, 255, 255];
pub const BLACK: Color = [0, 0, 0];
pub const YELLOW: Color = [255, 255, 0];
pub const CYAN: Color = [0, 255, 255];
pub const MAGENTA: Color = [255, 0, 255];
pub const ORANGE: Color = [255, 165, 0];

// ── Marker types ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub enum MarkerType {
    Cross,
    TiltedCross,
    Star,
    Diamond,
    Square,
    TriangleUp,
    TriangleDown,
}

// ── Internal helpers ────────────────────────────────────────────────────────

#[inline]
fn put_pixel_safe(img: &mut RgbImage, x: i32, y: i32, color: Color) {
    if x >= 0 && y >= 0 && (x as u32) < img.width() && (y as u32) < img.height() {
        img.put_pixel(x as u32, y as u32, Rgb(color));
    }
}

#[inline]
fn put_pixel_gray_safe(img: &mut GrayImage, x: i32, y: i32, val: u8) {
    if x >= 0 && y >= 0 && (x as u32) < img.width() && (y as u32) < img.height() {
        img.put_pixel(x as u32, y as u32, image::Luma([val]));
    }
}

fn draw_line_thin(img: &mut RgbImage, x0: i32, y0: i32, x1: i32, y1: i32, color: Color) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut x = x0;
    let mut y = y0;

    loop {
        put_pixel_safe(img, x, y, color);
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            if x == x1 {
                break;
            }
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            if y == y1 {
                break;
            }
            err += dx;
            y += sy;
        }
    }
}

fn draw_hline(img: &mut RgbImage, x0: i32, x1: i32, y: i32, color: Color) {
    let (lo, hi) = if x0 <= x1 { (x0, x1) } else { (x1, x0) };
    for x in lo..=hi {
        put_pixel_safe(img, x, y, color);
    }
}

// ── Public drawing functions ────────────────────────────────────────────────

/// Draw a line segment. Thickness -1 is treated as 1.
pub fn draw_line(img: &mut RgbImage, p1: (i32, i32), p2: (i32, i32), color: Color, thickness: i32) {
    let t = thickness.max(1);
    if t == 1 {
        draw_line_thin(img, p1.0, p1.1, p2.0, p2.1, color);
    } else {
        // Thick line: draw parallel offset lines
        let dx = (p2.0 - p1.0) as f32;
        let dy = (p2.1 - p1.1) as f32;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-6 {
            draw_circle(img, p1, t / 2, color, -1);
            return;
        }
        let nx = -dy / len;
        let ny = dx / len;
        let half = (t - 1) as f32 / 2.0;
        for i in 0..t {
            let offset = -half + i as f32;
            let ox = (nx * offset).round() as i32;
            let oy = (ny * offset).round() as i32;
            draw_line_thin(img, p1.0 + ox, p1.1 + oy, p2.0 + ox, p2.1 + oy, color);
        }
    }
}

/// Draw a rectangle. Thickness -1 = filled.
pub fn draw_rectangle(
    img: &mut RgbImage,
    p1: (i32, i32),
    p2: (i32, i32),
    color: Color,
    thickness: i32,
) {
    if thickness < 0 {
        // Filled
        let (y0, y1) = if p1.1 <= p2.1 {
            (p1.1, p2.1)
        } else {
            (p2.1, p1.1)
        };
        for y in y0..=y1 {
            draw_hline(img, p1.0, p2.0, y, color);
        }
    } else {
        draw_line(img, p1, (p2.0, p1.1), color, thickness);
        draw_line(img, (p2.0, p1.1), p2, color, thickness);
        draw_line(img, p2, (p1.0, p2.1), color, thickness);
        draw_line(img, (p1.0, p2.1), p1, color, thickness);
    }
}

/// Draw a circle. Thickness -1 = filled.
pub fn draw_circle(
    img: &mut RgbImage,
    center: (i32, i32),
    radius: i32,
    color: Color,
    thickness: i32,
) {
    if radius <= 0 {
        put_pixel_safe(img, center.0, center.1, color);
        return;
    }

    if thickness < 0 {
        // Filled circle via scanlines
        for y in -radius..=radius {
            let half_w = ((radius * radius - y * y) as f32).sqrt() as i32;
            draw_hline(
                img,
                center.0 - half_w,
                center.0 + half_w,
                center.1 + y,
                color,
            );
        }
    } else {
        // Midpoint circle algorithm
        let t = thickness.max(1);
        let r_outer = radius;
        let r_inner = (radius - t + 1).max(0);
        for y in -r_outer..=r_outer {
            for x in -r_outer..=r_outer {
                let d2 = x * x + y * y;
                if d2 <= r_outer * r_outer && d2 >= r_inner * r_inner {
                    put_pixel_safe(img, center.0 + x, center.1 + y, color);
                }
            }
        }
    }
}

/// Draw an ellipse. Thickness -1 = filled.
pub fn draw_ellipse(
    img: &mut RgbImage,
    center: (i32, i32),
    axes: (i32, i32),
    angle_deg: f32,
    color: Color,
    thickness: i32,
) {
    let (a, b) = (axes.0.max(1) as f32, axes.1.max(1) as f32);
    let cos_a = angle_deg.to_radians().cos();
    let sin_a = angle_deg.to_radians().sin();
    let max_r = a.max(b) as i32 + 1;

    if thickness < 0 {
        // Filled: test every pixel in bounding box
        for dy in -max_r..=max_r {
            for dx in -max_r..=max_r {
                let rx = dx as f32 * cos_a + dy as f32 * sin_a;
                let ry = -dx as f32 * sin_a + dy as f32 * cos_a;
                if (rx / a) * (rx / a) + (ry / b) * (ry / b) <= 1.0 {
                    put_pixel_safe(img, center.0 + dx, center.1 + dy, color);
                }
            }
        }
    } else {
        // Outline: sample points on ellipse perimeter
        let t = thickness.max(1);
        let circumference =
            std::f32::consts::PI * (3.0 * (a + b) - ((3.0 * a + b) * (a + 3.0 * b)).sqrt());
        let n_steps = (circumference * 2.0).max(100.0) as usize;
        let mut prev = None;
        for i in 0..=n_steps {
            let theta = 2.0 * std::f32::consts::PI * i as f32 / n_steps as f32;
            let ex = a * theta.cos();
            let ey = b * theta.sin();
            let px = center.0 + (ex * cos_a - ey * sin_a).round() as i32;
            let py = center.1 + (ex * sin_a + ey * cos_a).round() as i32;
            if let Some((px0, py0)) = prev {
                draw_line(img, (px0, py0), (px, py), color, t);
            }
            prev = Some((px, py));
        }
    }
}

/// Draw a polyline (connected line segments). If `closed`, connects last to first.
pub fn draw_polyline(
    img: &mut RgbImage,
    points: &[(i32, i32)],
    closed: bool,
    color: Color,
    thickness: i32,
) {
    if points.len() < 2 {
        return;
    }
    for i in 0..points.len() - 1 {
        draw_line(img, points[i], points[i + 1], color, thickness);
    }
    if closed && points.len() > 2 {
        draw_line(img, points[points.len() - 1], points[0], color, thickness);
    }
}

/// Draw a filled polygon using scanline fill.
pub fn draw_polygon(img: &mut RgbImage, points: &[(i32, i32)], color: Color) {
    if points.len() < 3 {
        return;
    }

    let min_y = points.iter().map(|p| p.1).min().unwrap();
    let max_y = points.iter().map(|p| p.1).max().unwrap();

    for y in min_y..=max_y {
        let mut intersections = Vec::new();
        let n = points.len();
        for i in 0..n {
            let j = (i + 1) % n;
            let (y0, y1) = (points[i].1, points[j].1);
            let (x0, x1) = (points[i].0, points[j].0);
            if (y0 <= y && y1 > y) || (y1 <= y && y0 > y) {
                let t = (y - y0) as f32 / (y1 - y0) as f32;
                intersections.push((x0 as f32 + t * (x1 - x0) as f32).round() as i32);
            }
        }
        intersections.sort();
        for pair in intersections.chunks(2) {
            if pair.len() == 2 {
                draw_hline(img, pair[0], pair[1], y, color);
            }
        }
    }
}

/// Draw an arrow from p1 to p2 with arrowhead.
pub fn draw_arrow(
    img: &mut RgbImage,
    p1: (i32, i32),
    p2: (i32, i32),
    color: Color,
    thickness: i32,
    tip_length: f32,
) {
    draw_line(img, p1, p2, color, thickness);

    let dx = (p2.0 - p1.0) as f32;
    let dy = (p2.1 - p1.1) as f32;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-6 {
        return;
    }

    let tip = tip_length * len;
    let angle: f32 = 0.5; // ~30 degrees
    let ux = dx / len;
    let uy = dy / len;
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    let lx = p2.0 - (tip * (ux * cos_a + uy * sin_a)) as i32;
    let ly = p2.1 - (tip * (-ux * sin_a + uy * cos_a)) as i32;
    let rx = p2.0 - (tip * (ux * cos_a - uy * sin_a)) as i32;
    let ry = p2.1 - (tip * (ux * sin_a + uy * cos_a)) as i32;

    draw_line(img, p2, (lx, ly), color, thickness);
    draw_line(img, p2, (rx, ry), color, thickness);
}

/// Draw a marker symbol at a position.
pub fn draw_marker(
    img: &mut RgbImage,
    pos: (i32, i32),
    marker: MarkerType,
    color: Color,
    size: i32,
) {
    let s = size / 2;
    match marker {
        MarkerType::Cross => {
            draw_line(img, (pos.0 - s, pos.1), (pos.0 + s, pos.1), color, 1);
            draw_line(img, (pos.0, pos.1 - s), (pos.0, pos.1 + s), color, 1);
        }
        MarkerType::TiltedCross => {
            draw_line(
                img,
                (pos.0 - s, pos.1 - s),
                (pos.0 + s, pos.1 + s),
                color,
                1,
            );
            draw_line(
                img,
                (pos.0 + s, pos.1 - s),
                (pos.0 - s, pos.1 + s),
                color,
                1,
            );
        }
        MarkerType::Star => {
            draw_marker(img, pos, MarkerType::Cross, color, size);
            draw_marker(img, pos, MarkerType::TiltedCross, color, size);
        }
        MarkerType::Diamond => {
            let pts = [
                (pos.0, pos.1 - s),
                (pos.0 + s, pos.1),
                (pos.0, pos.1 + s),
                (pos.0 - s, pos.1),
            ];
            draw_polyline(img, &pts, true, color, 1);
        }
        MarkerType::Square => {
            draw_rectangle(
                img,
                (pos.0 - s, pos.1 - s),
                (pos.0 + s, pos.1 + s),
                color,
                1,
            );
        }
        MarkerType::TriangleUp => {
            let pts = [
                (pos.0, pos.1 - s),
                (pos.0 + s, pos.1 + s),
                (pos.0 - s, pos.1 + s),
            ];
            draw_polyline(img, &pts, true, color, 1);
        }
        MarkerType::TriangleDown => {
            let pts = [
                (pos.0, pos.1 + s),
                (pos.0 + s, pos.1 - s),
                (pos.0 - s, pos.1 - s),
            ];
            draw_polyline(img, &pts, true, color, 1);
        }
    }
}

/// Draw text using built-in 8x8 bitmap font. `scale` multiplies the font size.
pub fn draw_text(img: &mut RgbImage, text: &str, origin: (i32, i32), scale: u32, color: Color) {
    let scale = scale.max(1);
    let mut cursor_x = origin.0;

    for ch in text.chars() {
        let idx = ch as u32;
        if idx < 32 || idx > 126 {
            cursor_x += 8 * scale as i32;
            continue;
        }
        let glyph = &drawing_font::FONT_8X8[(idx - 32) as usize];

        for row in 0..8u32 {
            let byte = glyph[row as usize];
            for col in 0..8u32 {
                if byte & (0x80 >> col) != 0 {
                    for sy in 0..scale {
                        for sx in 0..scale {
                            put_pixel_safe(
                                img,
                                cursor_x + (col * scale + sx) as i32,
                                origin.1 + (row * scale + sy) as i32,
                                color,
                            );
                        }
                    }
                }
            }
        }
        cursor_x += 8 * scale as i32;
    }
}

/// Draw text on a grayscale image.
pub fn draw_text_gray(img: &mut GrayImage, text: &str, origin: (i32, i32), scale: u32, value: u8) {
    let scale = scale.max(1);
    let mut cursor_x = origin.0;

    for ch in text.chars() {
        let idx = ch as u32;
        if idx < 32 || idx > 126 {
            cursor_x += 8 * scale as i32;
            continue;
        }
        let glyph = &drawing_font::FONT_8X8[(idx - 32) as usize];

        for row in 0..8u32 {
            let byte = glyph[row as usize];
            for col in 0..8u32 {
                if byte & (0x80 >> col) != 0 {
                    for sy in 0..scale {
                        for sx in 0..scale {
                            put_pixel_gray_safe(
                                img,
                                cursor_x + (col * scale + sx) as i32,
                                origin.1 + (row * scale + sy) as i32,
                                value,
                            );
                        }
                    }
                }
            }
        }
        cursor_x += 8 * scale as i32;
    }
}

// ── Convenience: feature visualization ──────────────────────────────────────

/// Draw keypoints as circles on an image.
pub fn draw_keypoints(img: &mut RgbImage, keypoints: &[cv_core::KeyPointF32], color: Color) {
    for kp in keypoints {
        let r = (kp.size / 2.0).max(2.0) as i32;
        draw_circle(img, (kp.x as i32, kp.y as i32), r, color, 1);
    }
}

/// Draw feature matches between two images side by side. Returns a new image.
pub fn draw_matches(
    img1: &GrayImage,
    kp1: &[cv_core::KeyPointF32],
    img2: &GrayImage,
    kp2: &[cv_core::KeyPointF32],
    matches: &[(usize, usize)],
    color: Color,
) -> RgbImage {
    let w1 = img1.width();
    let h = img1.height().max(img2.height());
    let total_w = w1 + img2.width();

    let mut out = RgbImage::new(total_w, h);

    // Copy img1
    for (x, y, p) in img1.enumerate_pixels() {
        out.put_pixel(x, y, Rgb([p.0[0], p.0[0], p.0[0]]));
    }
    // Copy img2 offset
    for (x, y, p) in img2.enumerate_pixels() {
        out.put_pixel(x + w1, y, Rgb([p.0[0], p.0[0], p.0[0]]));
    }

    // Draw matches
    for &(i, j) in matches {
        if i < kp1.len() && j < kp2.len() {
            let p1 = (kp1[i].x as i32, kp1[i].y as i32);
            let p2 = (kp2[j].x as i32 + w1 as i32, kp2[j].y as i32);
            draw_circle(&mut out, p1, 3, color, 1);
            draw_circle(&mut out, p2, 3, color, 1);
            draw_line(&mut out, p1, p2, color, 1);
        }
    }

    out
}

/// Draw contours on an image.
pub fn draw_contours(
    img: &mut RgbImage,
    contours: &[Vec<(i32, i32)>],
    color: Color,
    thickness: i32,
) {
    for contour in contours {
        draw_polyline(img, contour, true, color, thickness);
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_draw_line_basic() {
        let mut img = RgbImage::new(100, 100);
        draw_line(&mut img, (0, 0), (99, 99), RED, 1);
        // Diagonal should have red pixels
        assert_eq!(img.get_pixel(0, 0), &Rgb(RED));
        assert_eq!(img.get_pixel(50, 50), &Rgb(RED));
        assert_eq!(img.get_pixel(99, 99), &Rgb(RED));
    }

    #[test]
    fn test_draw_line_out_of_bounds() {
        let mut img = RgbImage::new(50, 50);
        // Should not panic
        draw_line(&mut img, (-100, -100), (200, 200), RED, 1);
    }

    #[test]
    fn test_draw_rectangle_filled() {
        let mut img = RgbImage::new(100, 100);
        draw_rectangle(&mut img, (10, 10), (20, 20), GREEN, -1);
        assert_eq!(img.get_pixel(15, 15), &Rgb(GREEN));
        assert_eq!(img.get_pixel(5, 5), &Rgb(BLACK));
    }

    #[test]
    fn test_draw_circle_filled() {
        let mut img = RgbImage::new(100, 100);
        draw_circle(&mut img, (50, 50), 10, BLUE, -1);
        assert_eq!(img.get_pixel(50, 50), &Rgb(BLUE)); // center
        assert_eq!(img.get_pixel(0, 0), &Rgb(BLACK)); // outside
    }

    #[test]
    fn test_draw_circle_outline() {
        let mut img = RgbImage::new(100, 100);
        draw_circle(&mut img, (50, 50), 20, RED, 1);
        // Center should NOT be red (outline only)
        assert_ne!(img.get_pixel(50, 50), &Rgb(RED));
        // Edge should be red
        assert_eq!(img.get_pixel(50, 30), &Rgb(RED));
    }

    #[test]
    fn test_draw_text() {
        let mut img = RgbImage::new(200, 50);
        draw_text(&mut img, "Hi", (0, 0), 1, WHITE);
        // Should have some white pixels in the first 16x8 area
        let mut found_white = false;
        for y in 0..8 {
            for x in 0..16 {
                if img.get_pixel(x, y) == &Rgb(WHITE) {
                    found_white = true;
                }
            }
        }
        assert!(found_white, "Text should produce visible pixels");
    }

    #[test]
    fn test_draw_polygon_filled() {
        let mut img = RgbImage::new(100, 100);
        let pts = [(50, 10), (90, 90), (10, 90)];
        draw_polygon(&mut img, &pts, YELLOW);
        // Center of triangle should be filled
        assert_eq!(img.get_pixel(50, 60), &Rgb(YELLOW));
    }

    #[test]
    fn test_draw_matches_creates_image() {
        let img1 = GrayImage::new(100, 100);
        let img2 = GrayImage::new(100, 100);
        let kp1 = vec![cv_core::KeyPointF32 {
            x: 50.0,
            y: 50.0,
            size: 7.0,
            angle: 0.0,
            response: 1.0,
            octave: 0,
            class_id: 0,
            padding: 0,
        }];
        let kp2 = kp1.clone();
        let matches = vec![(0, 0)];
        let out = draw_matches(&img1, &kp1, &img2, &kp2, &matches, GREEN);
        assert_eq!(out.width(), 200);
        assert_eq!(out.height(), 100);
    }
}
