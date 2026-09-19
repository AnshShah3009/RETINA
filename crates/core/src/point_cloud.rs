use nalgebra::{Point3, Scalar, Vector3};

#[derive(Debug, Clone, Default)]
pub struct PointCloud<T: Scalar = f32> {
    pub points: Vec<Point3<T>>,
    pub colors: Option<Vec<Point3<T>>>,
    pub normals: Option<Vec<Vector3<T>>>,
}

impl<T: Scalar> PointCloud<T> {
    pub fn new(points: Vec<Point3<T>>) -> Self {
        Self {
            points,
            colors: None,
            normals: None,
        }
    }

    pub fn with_colors(mut self, colors: Vec<Point3<T>>) -> crate::Result<Self> {
        if colors.len() == self.points.len() {
            self.colors = Some(colors);
            Ok(self)
        } else {
            Err(crate::Error::InvalidInput(format!(
                "Color count {} does not match point count {}",
                colors.len(),
                self.points.len()
            )))
        }
    }

    pub fn with_normals(mut self, normals: Vec<Vector3<T>>) -> crate::Result<Self> {
        if normals.len() == self.points.len() {
            self.normals = Some(normals);
            Ok(self)
        } else {
            Err(crate::Error::InvalidInput(format!(
                "Normal count {} does not match point count {}",
                normals.len(),
                self.points.len()
            )))
        }
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

pub type PointCloudf32 = PointCloud<f32>;
pub type PointCloudf64 = PointCloud<f64>;
