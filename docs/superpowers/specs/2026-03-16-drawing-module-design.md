# Drawing Module Design

## Context

RETINA is missing OpenCV-equivalent drawing functions (line, circle, rectangle, text, contours). Existing drawing code is private in `crates/examples/src/visualization_utils.rs`. Users of cv-features, cv-calib3d, and cv-3d have no way to annotate images through the public API.

## Location

`cv-imgproc/src/drawing.rs` — new module in the existing imgproc crate.

**Why imgproc**: cv-features and cv-calib3d already depend on cv-imgproc, so no new dependency edges. Matches OpenCV's organization.

## Image Types

Operates on `image` crate types (`RgbImage`, `GrayImage`). RETINA already has conversion between these and `Tensor` types.

## Color & Thickness Convention

```rust
/// RGB color as [u8; 3]. Named constants provided.
pub type Color = [u8; 3];

pub const RED: Color = [255, 0, 0];
pub const GREEN: Color = [0, 255, 0];
pub const BLUE: Color = [0, 0, 255];
pub const WHITE: Color = [255, 255, 255];
pub const BLACK: Color = [0, 0, 0];
pub const YELLOW: Color = [255, 255, 0];
pub const CYAN: Color = [0, 255, 255];
pub const MAGENTA: Color = [255, 0, 255];

/// Thickness: positive = outline width, -1 = filled (OpenCV convention)
```

## Public API

### Primitives

| Function | Signature |
|----------|-----------|
| `draw_line` | `(img: &mut RgbImage, p1: (i32, i32), p2: (i32, i32), color: Color, thickness: i32)` |
| `draw_rectangle` | `(img: &mut RgbImage, p1: (i32, i32), p2: (i32, i32), color: Color, thickness: i32)` |
| `draw_circle` | `(img: &mut RgbImage, center: (i32, i32), radius: i32, color: Color, thickness: i32)` |
| `draw_ellipse` | `(img: &mut RgbImage, center: (i32, i32), axes: (i32, i32), angle_deg: f32, color: Color, thickness: i32)` |
| `draw_polyline` | `(img: &mut RgbImage, points: &[(i32, i32)], closed: bool, color: Color, thickness: i32)` |
| `draw_polygon` | `(img: &mut RgbImage, points: &[(i32, i32)], color: Color, thickness: i32)` — always filled |
| `draw_arrow` | `(img: &mut RgbImage, p1: (i32, i32), p2: (i32, i32), color: Color, thickness: i32, tip_length: f32)` |
| `draw_marker` | `(img: &mut RgbImage, pos: (i32, i32), marker: MarkerType, color: Color, size: i32)` |

### Text

| Function | Signature |
|----------|-----------|
| `draw_text` | `(img: &mut RgbImage, text: &str, origin: (i32, i32), scale: u32, color: Color)` |

Built-in 8x8 bitmap font. No external font dependency. `scale` is pixel multiplier (1 = 8px tall, 2 = 16px tall).

### Convenience (feature visualization)

| Function | Signature |
|----------|-----------|
| `draw_keypoints` | `(img: &mut RgbImage, keypoints: &[KeyPointF32], color: Color)` |
| `draw_matches` | `(img1: &GrayImage, kp1: &[KeyPointF32], img2: &GrayImage, kp2: &[KeyPointF32], matches: &[(usize, usize)], color: Color) -> RgbImage` |
| `draw_contours` | `(img: &mut RgbImage, contours: &[Vec<(i32, i32)>], color: Color, thickness: i32)` |

### MarkerType enum

```rust
pub enum MarkerType {
    Cross,       // +
    TiltedCross, // x
    Star,        // + overlaid with x
    Diamond,
    Square,
    TriangleUp,
    TriangleDown,
}
```

## Algorithms

- **Line**: Bresenham's line algorithm with Wu's anti-aliasing for thickness > 1
- **Circle**: Midpoint circle algorithm, filled via horizontal scanlines
- **Ellipse**: Midpoint ellipse with rotation via coordinate transform
- **Polygon fill**: Scanline fill algorithm
- **Thick lines**: Offset parallel lines + round caps (Bresenham at each endpoint)
- **Text**: Built-in 8x8 ASCII bitmap font (chars 32-126), stored as `[u8; 95 * 8]` constant

## Files

- `crates/imgproc/src/drawing.rs` — all drawing functions
- `crates/imgproc/src/drawing_font.rs` — 8x8 bitmap font data
- Add `pub mod drawing;` to `crates/imgproc/src/lib.rs`

## Testing

- Unit tests for each primitive (draw on known-size image, check specific pixels)
- Bounds checking: drawing outside image bounds must not panic
- Thickness variants: outline vs filled
- Text rendering: verify character placement

## Verification

```
cargo test -p cv-imgproc -- drawing
cargo clippy -p cv-imgproc
```
