use super::contours::Contour;

/// Image moments up to 3rd order, including central moments.
#[derive(Debug, Clone, PartialEq)]
pub struct Moments {
    /// Spatial moment m00 (area for a filled contour).
    pub m00: f64,
    pub m10: f64,
    pub m01: f64,
    pub m20: f64,
    pub m11: f64,
    pub m02: f64,
    pub m30: f64,
    pub m21: f64,
    pub m12: f64,
    pub m03: f64,
    /// Central moments.
    pub mu20: f64,
    pub mu11: f64,
    pub mu02: f64,
    pub mu30: f64,
    pub mu21: f64,
    pub mu12: f64,
    pub mu03: f64,
}

impl Moments {
    /// Area of the contour (from m00 spatial moment).
    pub fn area(&self) -> f64 {
        self.m00
    }

    /// Centroid (center of mass) of the contour.
    /// Returns (x_bar, y_bar) or (0.0, 0.0) for zero-area contours.
    pub fn centroid(&self) -> (f64, f64) {
        if self.m00.abs() > 1e-12 {
            (self.m10 / self.m00, self.m01 / self.m00)
        } else {
            (0.0, 0.0)
        }
    }

    /// Compute the 7 Hu invariant moments.
    ///
    /// These are translation-, scale-, and rotation-invariant descriptors
    /// derived from normalized central moments up to 3rd order.
    pub fn hu_moments(&self) -> [f64; 7] {
        let m00 = self.m00;
        if m00.abs() < 1e-12 {
            return [0.0; 7];
        }

        let eta20 = self.mu20 / m00.powi(2);
        let eta11 = self.mu11 / m00.powi(2);
        let eta02 = self.mu02 / m00.powi(2);
        let eta30 = self.mu30 / m00.powf(2.5);
        let eta21 = self.mu21 / m00.powf(2.5);
        let eta12 = self.mu12 / m00.powf(2.5);
        let eta03 = self.mu03 / m00.powf(2.5);

        let h1 = eta20 + eta02;

        let h2 = (eta20 - eta02).powi(2) + 4.0 * eta11.powi(2);

        let h3 = (eta30 - 3.0 * eta12).powi(2) + (3.0 * eta21 - eta03).powi(2);

        let h4 = (eta30 + eta12).powi(2) + (eta21 + eta03).powi(2);

        let h5 = (eta30 - 3.0 * eta12) * (eta30 + eta12)
            * ((eta30 + eta12).powi(2) - 3.0 * (eta21 + eta03).powi(2))
            + (3.0 * eta21 - eta03) * (eta21 + eta03)
                * (3.0 * (eta30 + eta12).powi(2) - (eta21 + eta03).powi(2));

        let h6 = (eta20 - eta02) * ((eta30 + eta12).powi(2) - (eta21 + eta03).powi(2))
            + 4.0 * eta11 * (eta30 + eta12) * (eta21 + eta03);

        let h7 = (3.0 * eta21 - eta03) * (eta30 + eta12)
            * ((eta30 + eta12).powi(2) - 3.0 * (eta21 + eta03).powi(2))
            - (eta30 - 3.0 * eta12)
                * (eta21 + eta03)
                * (3.0 * (eta30 + eta12).powi(2) - (eta21 + eta03).powi(2));

        [
            h1.abs(),
            h2.abs(),
            h3.abs(),
            h4.abs(),
            h5.abs(),
            h6.abs(),
            h7.abs(),
        ]
    }
}

/// Polygon area (shoelace). Contour should be ordered.
pub fn contour_area(contour: &Contour) -> f64 {
    let n = contour.points.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0f64;
    for i in 0..n {
        let (x0, y0) = contour.points[i];
        let (x1, y1) = contour.points[(i + 1) % n];
        area += x0 as f64 * y1 as f64 - x1 as f64 * y0 as f64;
    }
    area.abs() * 0.5
}

/// Closed-contour perimeter.
pub fn contour_perimeter(contour: &Contour) -> f64 {
    let n = contour.points.len();
    if n < 2 {
        return 0.0;
    }
    let mut p = 0.0f64;
    for i in 0..n {
        let (x0, y0) = contour.points[i];
        let (x1, y1) = contour.points[(i + 1) % n];
        let dx = (x1 - x0) as f64;
        let dy = (y1 - y0) as f64;
        p += (dx * dx + dy * dy).sqrt();
    }
    p
}

/// Compute image moments of a contour up to 3rd order.
///
/// Uses the Green's theorem formulation for polygon moments.
/// The contour is treated as a closed polygon.
pub fn moments(contour: &Contour) -> Moments {
    let n = contour.points.len();
    let zero = Moments {
        m00: 0.0,
        m10: 0.0,
        m01: 0.0,
        m20: 0.0,
        m11: 0.0,
        m02: 0.0,
        m30: 0.0,
        m21: 0.0,
        m12: 0.0,
        m03: 0.0,
        mu20: 0.0,
        mu11: 0.0,
        mu02: 0.0,
        mu30: 0.0,
        mu21: 0.0,
        mu12: 0.0,
        mu03: 0.0,
    };

    if n < 3 {
        return zero;
    }

    // Compute raw spatial moments using Green's theorem for polygons
    let mut m00 = 0.0f64;
    let mut m10 = 0.0f64;
    let mut m01 = 0.0f64;
    let mut m20 = 0.0f64;
    let mut m11 = 0.0f64;
    let mut m02 = 0.0f64;
    let mut m30 = 0.0f64;
    let mut m21 = 0.0f64;
    let mut m12 = 0.0f64;
    let mut m03 = 0.0f64;

    for i in 0..n {
        let (xi, yi) = (contour.points[i].0 as f64, contour.points[i].1 as f64);
        let j = (i + 1) % n;
        let (xj, yj) = (contour.points[j].0 as f64, contour.points[j].1 as f64);
        let a = xi * yj - xj * yi; // cross product term

        m00 += a;
        m10 += a * (xi + xj);
        m01 += a * (yi + yj);
        m20 += a * (xi * xi + xi * xj + xj * xj);
        m11 += a * (2.0 * xi * yi + xi * yj + xj * yi + 2.0 * xj * yj);
        m02 += a * (yi * yi + yi * yj + yj * yj);
        m30 += a * (xi + xj) * (xi * xi + xj * xj);
        m21 +=
            a * (xi * xi * (3.0 * yi + yj) + 2.0 * xi * xj * (yi + yj) + xj * xj * (yi + 3.0 * yj));
        m12 +=
            a * (yi * yi * (3.0 * xi + xj) + 2.0 * yi * yj * (xi + xj) + yj * yj * (xi + 3.0 * xj));
        m03 += a * (yi + yj) * (yi * yi + yj * yj);
    }

    m00 /= 2.0;
    m10 /= 6.0;
    m01 /= 6.0;
    m20 /= 12.0;
    m11 /= 24.0;
    m02 /= 12.0;
    m30 /= 20.0;
    m21 /= 60.0;
    m12 /= 60.0;
    m03 /= 20.0;

    // Make moments positive (orientation-independent)
    if m00 < 0.0 {
        m00 = -m00;
        m10 = -m10;
        m01 = -m01;
        m20 = -m20;
        m11 = -m11;
        m02 = -m02;
        m30 = -m30;
        m21 = -m21;
        m12 = -m12;
        m03 = -m03;
    }

    // Central moments
    let x_bar = if m00.abs() > 1e-12 { m10 / m00 } else { 0.0 };
    let y_bar = if m00.abs() > 1e-12 { m01 / m00 } else { 0.0 };

    let mu20 = m20 - x_bar * m10;
    let mu11 = m11 - x_bar * m01;
    let mu02 = m02 - y_bar * m01;
    let mu30 = m30 - 3.0 * x_bar * m20 + 2.0 * x_bar * x_bar * m10;
    let mu21 = m21 - 2.0 * x_bar * m11 - y_bar * m20 + 2.0 * x_bar * x_bar * m01;
    let mu12 = m12 - 2.0 * y_bar * m11 - x_bar * m02 + 2.0 * y_bar * y_bar * m10;
    let mu03 = m03 - 3.0 * y_bar * m02 + 2.0 * y_bar * y_bar * m01;

    Moments {
        m00,
        m10,
        m01,
        m20,
        m11,
        m02,
        m30,
        m21,
        m12,
        m03,
        mu20,
        mu11,
        mu02,
        mu30,
        mu21,
        mu12,
        mu03,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moments_rectangle() {
        // Rectangle with vertices: (0,0), (4,0), (4,3), (0,3)
        let c = Contour {
            points: vec![(0, 0), (4, 0), (4, 3), (0, 3)],
        };
        let m = moments(&c);
        // m00 = area = 12
        assert!((m.m00 - 12.0).abs() < 1e-6);
        // Centroid at (2, 1.5)
        let cx = m.m10 / m.m00;
        let cy = m.m01 / m.m00;
        assert!((cx - 2.0).abs() < 1e-6);
        assert!((cy - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_moments_triangle() {
        // Right triangle (0,0), (6,0), (0,4)
        let c = Contour {
            points: vec![(0, 0), (6, 0), (0, 4)],
        };
        let m = moments(&c);
        // Area = 0.5 * 6 * 4 = 12
        assert!((m.m00 - 12.0).abs() < 1e-6);
        // Centroid at (2, 4/3)
        let cx = m.m10 / m.m00;
        let cy = m.m01 / m.m00;
        assert!((cx - 2.0).abs() < 1e-6);
        assert!((cy - 4.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_centroid_method() {
        let c = Contour {
            points: vec![(0, 0), (4, 0), (4, 3), (0, 3)],
        };
        let m = moments(&c);
        let (cx, cy) = m.centroid();
        assert!((cx - 2.0).abs() < 1e-6);
        assert!((cy - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_hu_moments_square() {
        let c = Contour {
            points: vec![(0, 0), (10, 0), (10, 10), (0, 10)],
        };
        let m = moments(&c);
        let hu = m.hu_moments();
        // Hu moments should be valid (non-NaN, finite)
        for &h in &hu {
            assert!(h.is_finite());
        }
        // For a centered square, h1 should be ~0.166...
        assert!((hu[0] - 0.16666666666666666).abs() < 0.01);
        // h2 should be 0 for a symmetric square
        assert!(hu[1].abs() < 1e-12);
    }

    #[test]
    fn test_hu_moments_translation_invariant() {
        // Same shape at different positions should have same Hu moments
        let c1 = Contour {
            points: vec![(0, 0), (4, 0), (4, 3), (0, 3)],
        };
        let c2 = Contour {
            points: vec![(10, 20), (14, 20), (14, 23), (10, 23)],
        };
        let hu1 = moments(&c1).hu_moments();
        let hu2 = moments(&c2).hu_moments();
        for i in 0..7 {
            assert!(
                (hu1[i] - hu2[i]).abs() < 1e-6,
                "Hu moment {} differs: {} vs {}",
                i,
                hu1[i],
                hu2[i]
            );
        }
    }
}
