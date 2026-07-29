pub mod camera;
pub mod distortion;
pub mod pose;

pub use camera::*;
pub use distortion::*;
pub use pose::*;

pub type Vector6<T> = nalgebra::Vector6<T>;

/// A 2D axis-aligned rectangle.
#[derive(Debug, Clone, Copy, Default)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Rect {
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }

    pub fn x1(&self) -> f32 {
        self.x
    }
    pub fn y1(&self) -> f32 {
        self.y
    }
    pub fn x2(&self) -> f32 {
        self.x + self.w
    }
    pub fn y2(&self) -> f32 {
        self.y + self.h
    }

    pub fn area(&self) -> f32 {
        self.w * self.h
    }

    pub fn iou(&self, other: &Rect) -> f32 {
        let x1 = self.x1().max(other.x1());
        let y1 = self.y1().max(other.y1());
        let x2 = self.x2().min(other.x2());
        let y2 = self.y2().min(other.y2());

        let w = (x2 - x1).max(0.0);
        let h = (y2 - y1).max(0.0);
        let intersection = w * h;

        if intersection == 0.0 {
            return 0.0;
        }

        let union = self.area() + other.area() - intersection;
        intersection / union
    }
}

/// A 2D rotated rectangle defined by center, size, and angle.
#[derive(Debug, Clone, Copy, Default)]
pub struct RotatedRect {
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
    pub angle: f32, // Degrees
}

impl RotatedRect {
    pub fn new(cx: f32, cy: f32, w: f32, h: f32, angle: f32) -> Self {
        Self {
            cx,
            cy,
            w,
            h,
            angle,
        }
    }

    pub fn area(&self) -> f32 {
        self.w * self.h
    }

    /// Get the 4 corners of the rotated rectangle
    pub fn points(&self) -> [[f32; 2]; 4] {
        let angle_rad = self.angle.to_radians();
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();

        let half_w = self.w / 2.0;
        let half_h = self.h / 2.0;

        let mut pts = [[0.0, 0.0]; 4];
        // Relative corners before rotation
        let corners = [
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h],
        ];

        for i in 0..4 {
            pts[i][0] = self.cx + corners[i][0] * cos_a - corners[i][1] * sin_a;
            pts[i][1] = self.cy + corners[i][0] * sin_a + corners[i][1] * cos_a;
        }
        pts
    }
}

/// A generic 2D polygon defined by vertices.
#[derive(Debug, Clone, Default)]
pub struct Polygon {
    pub points: Vec<[f32; 2]>,
}

impl Polygon {
    pub fn new(points: Vec<[f32; 2]>) -> Self {
        Self { points }
    }

    pub fn area(&self) -> f32 {
        if self.points.len() < 3 {
            return 0.0;
        }
        let mut area = 0.0;
        for i in 0..self.points.len() {
            let p1 = self.points[i];
            let p2 = self.points[(i + 1) % self.points.len()];
            area += p1[0] * p2[1] - p2[0] * p1[1];
        }
        area * 0.5 // Keep signed area for winding check
    }

    pub fn is_clockwise(&self) -> bool {
        self.area() < 0.0
    }

    pub fn ensure_counter_clockwise(&mut self) {
        if self.is_clockwise() {
            self.points.reverse();
        }
    }

    pub fn unsigned_area(&self) -> f32 {
        self.area().abs()
    }
}

/// Calculates the Intersection over Union (IoU) of two rotated rectangles.
pub fn rotated_iou(r1: &RotatedRect, r2: &RotatedRect) -> f32 {
    let mut p1 = Polygon::new(r1.points().to_vec());
    let mut p2 = Polygon::new(r2.points().to_vec());
    p1.ensure_counter_clockwise();
    p2.ensure_counter_clockwise();
    polygon_iou(&p1, &p2)
}

/// Calculates the Intersection over Union (IoU) of two polygons.
pub fn polygon_iou(p1: &Polygon, p2: &Polygon) -> f32 {
    let inter_area = intersection_area_polygons(p1, p2);
    let a1 = p1.unsigned_area();
    let a2 = p2.unsigned_area();
    if inter_area <= 0.0 {
        return 0.0;
    }
    let union_area = a1 + a2 - inter_area;
    inter_area / union_area
}

/// Calculates the intersection area of two convex polygons using Sutherland-Hodgman clipping.
pub fn intersection_area_polygons(p1: &Polygon, p2: &Polygon) -> f32 {
    // Sutherland-Hodgman clipping for generic convex polygons
    let pts1 = &p1.points;
    let pts2 = &p2.points;

    if pts1.len() < 3 || pts2.len() < 3 {
        return 0.0;
    }

    let mut poly = pts1.clone();

    // Clip pts1 against each edge of pts2
    for i in 0..pts2.len() {
        let edge_p1 = pts2[i];
        let edge_p2 = pts2[(i + 1) % pts2.len()];

        let mut next_poly = Vec::new();
        if poly.is_empty() {
            return 0.0;
        }

        for j in 0..poly.len() {
            let cur = poly[j];
            let prev = poly[(j + poly.len() - 1) % poly.len()];

            let is_cur_inside = is_inside(edge_p1, edge_p2, cur);
            let is_prev_inside = is_inside(edge_p1, edge_p2, prev);

            if is_cur_inside {
                if !is_prev_inside {
                    next_poly.push(intersect(prev, cur, edge_p1, edge_p2));
                }
                next_poly.push(cur);
            } else if is_prev_inside {
                next_poly.push(intersect(prev, cur, edge_p1, edge_p2));
            }
        }
        poly = next_poly;
    }

    if poly.len() < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..poly.len() {
        let p1 = poly[i];
        let p2 = poly[(i + 1) % poly.len()];
        area += p1[0] * p2[1] - p2[0] * p1[1];
    }
    area.abs() * 0.5
}

fn is_inside(p1: [f32; 2], p2: [f32; 2], p: [f32; 2]) -> bool {
    (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0]) >= 0.0
}

fn intersect(a: [f32; 2], b: [f32; 2], c: [f32; 2], d: [f32; 2]) -> [f32; 2] {
    let a1 = b[1] - a[1];
    let b1 = a[0] - b[0];
    let c1 = a1 * a[0] + b1 * a[1];

    let a2 = d[1] - c[1];
    let b2 = c[0] - d[0];
    let c2 = a2 * c[0] + b2 * c[1];

    let det = a1 * b2 - a2 * b1;
    if det.abs() < 1e-6 {
        return a; // Parallel
    }
    [(b2 * c1 - b1 * c2) / det, (a1 * c2 - a2 * c1) / det]
}

/// A line in Hesse normal form (rho, theta).
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct HoughLine {
    /// Distance from the origin.
    pub rho: f32,
    /// Angle in radians.
    pub theta: f32,
    /// Accumulator score.
    pub score: u32,
}

impl HoughLine {
    pub fn new(rho: f32, theta: f32, score: u32) -> Self {
        Self { rho, theta, score }
    }
}

/// A circle in (x, y, radius) form.
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct HoughCircle {
    pub cx: f32,
    pub cy: f32,
    pub r: f32,
    pub score: u32,
}

impl HoughCircle {
    pub fn new(cx: f32, cy: f32, r: f32, score: u32) -> Self {
        Self { cx, cy, r, score }
    }
}

/// An object detection result.
#[derive(Debug, Clone, Copy, Default)]
pub struct Detection {
    /// Bounding box of the detection.
    pub rect: Rect,
    /// Confidence score (0.0 to 1.0).
    pub score: f32,
    /// Class identifier.
    pub class_id: i32,
}

impl Detection {
    pub fn new(rect: Rect, score: f32, class_id: i32) -> Self {
        Self {
            rect,
            score,
            class_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix3, Point3, UnitQuaternion, Vector3};

    mod pose_tests {
        use super::*;

        #[test]
        fn test_pose_identity() {
            let pose = Pose::identity();
            assert!((pose.rotation.as_ref() - UnitQuaternion::identity().as_ref()).norm() < 1e-10);
            assert_eq!(pose.translation, Vector3::zeros());
        }

        #[test]
        fn test_pose_new() {
            let rotation = Matrix3::identity();
            let translation = Vector3::new(1.0, 2.0, 3.0);
            let pose = Pose::new(rotation, translation);

            assert_eq!(pose.translation.x, 1.0);
            assert_eq!(pose.translation.y, 2.0);
            assert_eq!(pose.translation.z, 3.0);
        }

        #[test]
        fn test_pose_inverse_roundtrip() {
            let rotation = Matrix3::identity();
            let translation = Vector3::new(1.0, 2.0, 3.0);
            let pose = Pose::new(rotation, translation);

            let result = pose.compose(&pose.inverse());
            assert!(
                (result.rotation.as_ref() - UnitQuaternion::identity().as_ref()).norm() < 1e-10
            );
            assert!(result.translation.norm() < 1e-10);
        }

        #[test]
        fn test_pose_compose_identity() {
            let translation = Vector3::new(0.5, 0.6, 0.7);
            let pose = Pose::new(Matrix3::identity(), translation);

            let result = pose.compose(&Pose::identity());
            assert!((result.translation - translation).norm() < 1e-10);
        }

        #[test]
        fn test_pose_transform_point() {
            let pose = Pose::identity();
            let point = Point3::new(1.0, 2.0, 3.0);
            let result = pose.transform_point(&point);

            assert!((result - point).norm() < 1e-10);
        }

        #[test]
        fn test_pose_matrix() {
            let pose = Pose::identity();
            let matrix = pose.matrix();

            assert_eq!(matrix[(0, 0)], 1.0);
            assert_eq!(matrix[(1, 1)], 1.0);
            assert_eq!(matrix[(2, 2)], 1.0);
            assert_eq!(matrix[(3, 3)], 1.0);
        }

        #[test]
        fn test_pose_default() {
            let pose = Pose::default();
            assert!((pose.rotation.as_ref() - UnitQuaternion::identity().as_ref()).norm() < 1e-10);
            assert_eq!(pose.translation, Vector3::zeros());
        }
    }

    mod camera_intrinsics_tests {
        use super::*;

        #[test]
        fn test_camera_intrinsics_new() {
            let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);

            assert_eq!(intrinsics.fx, 500.0);
            assert_eq!(intrinsics.fy, 500.0);
            assert_eq!(intrinsics.cx, 320.0);
            assert_eq!(intrinsics.cy, 240.0);
        }

        #[test]
        fn test_camera_intrinsics_new_ideal() {
            let intrinsics = CameraIntrinsics::new_ideal(640, 480);

            assert_eq!(intrinsics.fx, 640.0);
            assert_eq!(intrinsics.fy, 640.0);
            assert_eq!(intrinsics.cx, 320.0);
            assert_eq!(intrinsics.cy, 240.0);
        }

        #[test]
        fn test_camera_intrinsics_f32() {
            let intrinsics = CameraIntrinsicsF32::new(500.0, 500.0, 320.0, 240.0, 640, 480);

            assert_eq!(intrinsics.fx, 500.0f32);
            assert_eq!(intrinsics.fy, 500.0f32);
        }
    }

    mod distortion_tests {
        use super::*;

        #[test]
        fn test_distortion_identity() {
            let dist = Distortion::new(0.0, 0.0, 0.0, 0.0, 0.0);

            let (xd, yd) = dist.apply(0.5, 0.5);
            assert!((xd - 0.5).abs() < 1e-10);
            assert!((yd - 0.5).abs() < 1e-10);
        }

        #[test]
        fn test_distortion_apply_remove_roundtrip() {
            let dist = Distortion::new(0.1, 0.01, 0.001, 0.001, 0.0);

            let (xd, yd) = dist.apply(0.3, 0.4);
            let (xr, yr) = dist.remove(xd, yd);

            assert!((xr - 0.3).abs() < 1e-4);
            assert!((yr - 0.4).abs() < 1e-4);
        }

        #[test]
        fn test_distortion_at_origin() {
            let dist = Distortion::new(0.5, 0.3, 0.01, 0.01, 0.1);

            let (xd, yd) = dist.apply(0.0, 0.0);
            assert!((xd).abs() < 1e-10);
            assert!((yd).abs() < 1e-10);
        }

        #[test]
        fn test_fisheye_distortion_identity() {
            let dist = FisheyeDistortion::new(0.0, 0.0, 0.0, 0.0);

            let (xd, yd) = dist.apply(0.0, 0.0);
            assert!((xd).abs() < 1e-10);
            assert!((yd).abs() < 1e-10);
        }

        #[test]
        fn test_fisheye_distortion_roundtrip() {
            let dist = FisheyeDistortion::new(0.1, 0.01, 0.001, 0.001);

            let (xd, yd) = dist.apply(0.3, 0.4);
            let (xr, yr) = dist.remove(xd, yd);

            assert!((xr - 0.3).abs() < 1e-3);
            assert!((yr - 0.4).abs() < 1e-3);
        }
    }

    mod rect_tests {
        use super::*;

        #[test]
        fn test_rect_area() {
            let rect = Rect::new(0.0, 0.0, 10.0, 20.0);
            assert!((rect.area() - 200.0).abs() < 1e-5);
        }

        #[test]
        fn test_rect_area_negative() {
            let rect = Rect::new(0.0, 0.0, 0.0, 20.0);
            assert!((rect.area() - 0.0).abs() < 1e-5);
        }

        #[test]
        fn test_rect_iou_identical() {
            let rect = Rect::new(0.0, 0.0, 10.0, 10.0);
            assert!((rect.iou(&rect) - 1.0).abs() < 1e-5);
        }

        #[test]
        fn test_rect_iou_no_overlap() {
            let rect1 = Rect::new(0.0, 0.0, 10.0, 10.0);
            let rect2 = Rect::new(20.0, 20.0, 10.0, 10.0);
            assert!((rect1.iou(&rect2)).abs() < 1e-5);
        }

        #[test]
        fn test_rect_iou_partial_overlap() {
            let rect1 = Rect::new(0.0, 0.0, 10.0, 10.0);
            let rect2 = Rect::new(5.0, 5.0, 10.0, 10.0);

            let iou = rect1.iou(&rect2);
            assert!(iou > 0.0 && iou < 1.0);
        }

        #[test]
        fn test_rect_bounds() {
            let rect = Rect::new(1.0, 2.0, 10.0, 20.0);

            assert!((rect.x1() - 1.0).abs() < 1e-5);
            assert!((rect.y1() - 2.0).abs() < 1e-5);
            assert!((rect.x2() - 11.0).abs() < 1e-5);
            assert!((rect.y2() - 22.0).abs() < 1e-5);
        }
    }

    mod polygon_tests {
        use super::*;

        fn create_square() -> Polygon {
            Polygon {
                points: vec![[0.0f32, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
            }
        }

        #[test]
        fn test_polygon_area_square() {
            let poly = create_square();
            let area = poly.unsigned_area();

            assert!((area - 100.0).abs() < 1e-5);
        }

        #[test]
        fn test_polygon_area_triangle() {
            let poly = Polygon {
                points: vec![[0.0f32, 0.0], [10.0, 0.0], [0.0, 10.0]],
            };

            assert!((poly.unsigned_area() - 50.0).abs() < 1e-5);
        }

        #[test]
        fn test_polygon_area_empty() {
            let poly = Polygon { points: vec![] };
            assert_eq!(poly.area(), 0.0);
        }

        #[test]
        fn test_polygon_iou_identical() {
            let poly = create_square();

            assert!((polygon_iou(&poly, &poly) - 1.0).abs() < 1e-5);
        }

        #[test]
        fn test_polygon_iou_no_overlap() {
            let poly1 = Polygon {
                points: vec![[0.0f32, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
            };
            let poly2 = Polygon {
                points: vec![[20.0f32, 20.0], [30.0, 20.0], [30.0, 30.0], [20.0, 30.0]],
            };

            assert!(polygon_iou(&poly1, &poly2).abs() < 1e-5);
        }
    }

    mod geometry_util_tests {
        use super::*;

        #[test]
        fn test_skew_symmetric_zero() {
            let v = Vector3::zeros();
            let skew = skew_symmetric(&v);

            assert_eq!(skew, Matrix3::zeros());
        }

        #[test]
        fn test_skew_symmetric_unit_x() {
            let v = Vector3::x();
            let skew = skew_symmetric(&v);

            assert_eq!(skew[(0, 1)], 0.0);
            assert_eq!(skew[(0, 2)], 0.0);
            assert_eq!(skew[(1, 0)], 0.0);
            assert_eq!(skew[(1, 2)], -1.0);
            assert_eq!(skew[(2, 0)], 0.0);
            assert_eq!(skew[(2, 1)], 1.0);
        }

        #[test]
        fn test_twist_to_se3_zero() {
            let twist = Vector6::zeros();
            let (r, t) = twist_to_se3(&twist);

            assert!(r.is_identity(1e-10));
            assert!(t.norm() < 1e-10);
        }

        #[test]
        fn test_twist_to_se3_rotation_90deg_z() {
            // Pure rotation of PI/2 about the Z axis.
            // twist = [omega_x, omega_y, omega_z, v_x, v_y, v_z]
            //       = [0, 0, PI/2, 0, 0, 0]
            let half_pi = std::f64::consts::FRAC_PI_2;
            let twist = Vector6::<f64>::new(0.0, 0.0, half_pi, 0.0, 0.0, 0.0);
            let (r, t) = twist_to_se3(&twist);

            // Expected rotation matrix for 90-degree rotation about Z:
            //   [ cos(90)  -sin(90)  0 ]     [ 0  -1  0 ]
            //   [ sin(90)   cos(90)  0 ]  =  [ 1   0  0 ]
            //   [    0         0     1 ]     [ 0   0  1 ]
            let eps = 1e-9;
            assert!(
                (r[(0, 0)] - 0.0).abs() < eps,
                "R[0,0] should be ~0, got {}",
                r[(0, 0)]
            );
            assert!(
                (r[(0, 1)] - (-1.0)).abs() < eps,
                "R[0,1] should be ~-1, got {}",
                r[(0, 1)]
            );
            assert!(
                (r[(0, 2)]).abs() < eps,
                "R[0,2] should be ~0, got {}",
                r[(0, 2)]
            );
            assert!(
                (r[(1, 0)] - 1.0).abs() < eps,
                "R[1,0] should be ~1, got {}",
                r[(1, 0)]
            );
            assert!(
                (r[(1, 1)] - 0.0).abs() < eps,
                "R[1,1] should be ~0, got {}",
                r[(1, 1)]
            );
            assert!(
                (r[(1, 2)]).abs() < eps,
                "R[1,2] should be ~0, got {}",
                r[(1, 2)]
            );
            assert!(
                (r[(2, 0)]).abs() < eps,
                "R[2,0] should be ~0, got {}",
                r[(2, 0)]
            );
            assert!(
                (r[(2, 1)]).abs() < eps,
                "R[2,1] should be ~0, got {}",
                r[(2, 1)]
            );
            assert!(
                (r[(2, 2)] - 1.0).abs() < eps,
                "R[2,2] should be ~1, got {}",
                r[(2, 2)]
            );

            // Pure rotation: translation should be zero
            assert!(
                t.norm() < eps,
                "Translation should be zero for pure rotation, got {:?}",
                t
            );
        }
    }

    mod rotated_rect_tests {
        use super::*;

        #[test]
        fn test_rotated_rect_area() {
            let rect = RotatedRect::new(10.0, 10.0, 20.0, 10.0, 0.0);
            assert!((rect.area() - 200.0).abs() < 1e-5);
        }

        #[test]
        fn test_rotated_rect_area_rotated() {
            let rect = RotatedRect::new(10.0, 10.0, 20.0, 10.0, 45.0);
            assert!((rect.area() - 200.0).abs() < 1e-5);
        }

        #[test]
        fn test_rotated_rect_points() {
            let rect = RotatedRect::new(50.0, 50.0, 10.0, 20.0, 0.0);
            let points = rect.points();
            assert_eq!(points.len(), 4);
        }
    }

    mod from_into_traits_tests {
        use super::*;

        #[test]
        fn test_camera_intrinsics_f64_to_f32() {
            let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
            let intrinsics_f32: CameraIntrinsicsF32 = intrinsics.into();
            assert!((intrinsics_f32.fx - 500.0).abs() < 1e-5);
            assert!((intrinsics_f32.fy - 500.0).abs() < 1e-5);
            assert!((intrinsics_f32.cx - 320.0).abs() < 1e-5);
            assert!((intrinsics_f32.cy - 240.0).abs() < 1e-5);
            assert_eq!(intrinsics_f32.width, 640);
            assert_eq!(intrinsics_f32.height, 480);
        }

        #[test]
        fn test_camera_intrinsics_f32_to_f64() {
            let intrinsics_f32 = CameraIntrinsicsF32::new(500.0, 500.0, 320.0, 240.0, 640, 480);
            let intrinsics: CameraIntrinsics = intrinsics_f32.into();
            assert!((intrinsics.fx - 500.0).abs() < 1e-10);
            assert!((intrinsics.fy - 500.0).abs() < 1e-10);
            assert!((intrinsics.cx - 320.0).abs() < 1e-10);
            assert!((intrinsics.cy - 240.0).abs() < 1e-10);
            assert_eq!(intrinsics.width, 640);
            assert_eq!(intrinsics.height, 480);
        }

        #[test]
        fn test_distortion_f64_to_f32() {
            let distortion = Distortion::new(0.1, 0.01, 0.001, 0.0001, 0.00001);
            let distortion_f32: DistortionF32 = distortion.into();
            assert!((distortion_f32.k1 - 0.1).abs() < 1e-5);
            assert!((distortion_f32.k2 - 0.01).abs() < 1e-5);
            assert!((distortion_f32.p1 - 0.001).abs() < 1e-5);
            assert!((distortion_f32.p2 - 0.0001).abs() < 1e-5);
            assert!((distortion_f32.k3 - 0.00001).abs() < 1e-6);
        }

        #[test]
        fn test_distortion_f32_to_f64() {
            let distortion_f32 = DistortionF32::new(0.1, 0.01, 0.001, 0.0001, 0.00001);
            let distortion: Distortion = distortion_f32.into();
            // f32→f64 conversions have ~1e-7 relative error for values ~0.1
            assert!((distortion.k1 - 0.1).abs() < 1e-7);
            assert!((distortion.k2 - 0.01).abs() < 1e-8);
            assert!((distortion.p1 - 0.001).abs() < 1e-8);
            assert!((distortion.p2 - 0.0001).abs() < 1e-8);
            assert!((distortion.k3 - 0.00001).abs() < 1e-9);
        }

        #[test]
        fn test_fisheye_distortion_f64_to_f32() {
            let fisheye = FisheyeDistortion::new(0.1, 0.01, 0.001, 0.0001);
            let fisheye_f32: FisheyeDistortionF32 = fisheye.into();
            assert!((fisheye_f32.k1 - 0.1).abs() < 1e-5);
            assert!((fisheye_f32.k2 - 0.01).abs() < 1e-5);
            assert!((fisheye_f32.k3 - 0.001).abs() < 1e-5);
            assert!((fisheye_f32.k4 - 0.0001).abs() < 1e-5);
        }

        #[test]
        fn test_fisheye_distortion_f32_to_f64() {
            let fisheye_f32 = FisheyeDistortionF32::new(0.1, 0.01, 0.001, 0.0001);
            let fisheye: FisheyeDistortion = fisheye_f32.into();
            // f32→f64 conversions have ~1e-7 relative error for values ~0.1
            assert!((fisheye.k1 - 0.1).abs() < 1e-7);
            assert!((fisheye.k2 - 0.01).abs() < 1e-8);
            assert!((fisheye.k3 - 0.001).abs() < 1e-8);
            assert!((fisheye.k4 - 0.0001).abs() < 1e-8);
        }
    }

    mod projection_tests {
        use super::*;

        #[test]
        fn test_project_point_without_extrinsics_or_distortion() {
            let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
            let point = Point3::new(0.1, 0.05, 1.0);

            let result = project_point(&point, &intrinsics, None, None);
            assert!(result.is_some());

            let proj = result.unwrap();
            // u = fx * x/z + cx = 500 * 0.1 + 320 = 370
            // v = fy * y/z + cy = 500 * 0.05 + 240 = 265
            assert!((proj.x - 370.0).abs() < 1e-10);
            assert!((proj.y - 265.0).abs() < 1e-10);
        }

        #[test]
        fn test_project_point_with_extrinsics() {
            let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
            let extrinsics = Pose::identity();
            let point = Point3::new(0.1, 0.05, 1.0);

            let result = project_point(&point, &intrinsics, Some(&extrinsics), None);
            assert!(result.is_some());

            let proj = result.unwrap();
            // With identity pose, should be same as without extrinsics
            assert!((proj.x - 370.0).abs() < 1e-10);
            assert!((proj.y - 265.0).abs() < 1e-10);
        }

        #[test]
        fn test_project_point_with_distortion() {
            let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
            let distortion = Distortion::new(0.0, 0.0, 0.0, 0.0, 0.0); // No distortion
            let point = Point3::new(0.1, 0.05, 1.0);

            let result = project_point(&point, &intrinsics, None, Some(&distortion));
            assert!(result.is_some());

            let proj = result.unwrap();
            // No distortion, should be same as without
            assert!((proj.x - 370.0).abs() < 1e-10);
            assert!((proj.y - 265.0).abs() < 1e-10);
        }

        #[test]
        fn test_project_point_guards_against_zero_depth() {
            let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
            let point = Point3::new(0.1, 0.05, 0.0);

            let result = project_point(&point, &intrinsics, None, None);
            assert!(result.is_none());
        }

        #[test]
        fn test_project_point_guards_against_negative_depth() {
            let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
            let point = Point3::new(0.1, 0.05, -1.0);

            let result = project_point(&point, &intrinsics, None, None);
            assert!(result.is_none());
        }

        #[test]
        fn test_project_point_guards_against_non_finite_coordinates() {
            let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
            let point = Point3::new(0.1, 0.05, f64::NAN);

            let result = project_point(&point, &intrinsics, None, None);
            assert!(result.is_none());
        }
    }
}
