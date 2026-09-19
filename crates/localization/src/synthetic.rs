//! A reproducible synthetic localization scene.
//!
//! [`generate`] builds a deterministic world: a cloud of 3D landmarks that each
//! carry a unique random binary descriptor, a ring of database views that observe
//! (and are given known poses for) a subset of those landmarks, and a query view
//! with a known pose that is slightly offset from one of the database views.
//!
//! Because the descriptors are exact copies of their landmark's descriptor, the
//! retrieval and matching layers see the same clean signal they would see on
//! well-textured real imagery, so the scene is a faithful end-to-end fixture for
//! the tests and for the bundled example. Everything is driven by a tiny
//! in-module xorshift RNG, so no external dependency is required and repeated
//! runs are bit-for-bit identical.

use crate::database::{Database, DatabaseImage, Landmark};
use cv_core::{CameraIntrinsics, Descriptor, Descriptors, KeyPoint, Pose};
use cv_features::retrieval::Vocabulary;
use nalgebra::{Matrix3, Point3, Rotation3, UnitQuaternion, Vector3};

/// Knobs for [`generate`]. [`SyntheticConfig::default`] matches the scene used
/// throughout the crate's tests: 200 landmarks and 8 database views.
#[derive(Debug, Clone, Copy)]
pub struct SyntheticConfig {
    /// Number of 3D landmarks.
    pub num_landmarks: usize,
    /// Number of database views.
    pub num_views: usize,
    /// Descriptor length in bytes.
    pub descriptor_dim: usize,
    /// Edge length of the cube the landmarks are drawn from.
    pub extent: f64,
    /// Radius of the ring the database cameras sit on.
    pub ring_radius: f64,
    /// Image width in pixels.
    pub image_width: u32,
    /// Image height in pixels.
    pub image_height: u32,
    /// Focal length (both axes) in pixels.
    pub focal: f64,
    /// Master RNG seed.
    pub seed: u64,
}

impl Default for SyntheticConfig {
    fn default() -> Self {
        Self {
            num_landmarks: 200,
            num_views: 8,
            descriptor_dim: 32,
            extent: 30.0,
            ring_radius: 18.0,
            image_width: 640,
            image_height: 480,
            focal: 500.0,
            seed: 0xC0FFEE,
        }
    }
}

/// A generated query view: the observations a user would extract from the image,
/// plus the pose they are trying to recover.
#[derive(Debug, Clone)]
pub struct Query {
    /// Query keypoints.
    pub keypoints: Vec<KeyPoint>,
    /// Query descriptors, parallel to `keypoints`.
    pub descriptors: Descriptors,
    /// Ground-truth world-to-camera pose.
    pub pose: Pose,
    /// Landmark observed by each keypoint.
    pub landmark_ids: Vec<usize>,
}

/// A generated scene: landmarks, database views and a query view.
pub struct SyntheticScene {
    /// Shared camera intrinsics.
    pub camera: CameraIntrinsics,
    /// The 3D landmarks.
    pub landmarks: Vec<Landmark>,
    /// Database views (ids `0..num_views`).
    pub views: Vec<DatabaseImage>,
    /// Landmark indices visible in each view, parallel to `views`.
    view_landmark_ids: Vec<Vec<usize>>,
    /// The query observations and their ground-truth pose.
    pub query: Query,
    /// Index (into `views`) of the database image with the largest landmark
    /// overlap with the query.
    correct_view: usize,
    /// Descriptor length in bytes.
    descriptor_dim: usize,
}

impl SyntheticScene {
    /// The shared camera intrinsics.
    pub fn camera(&self) -> &CameraIntrinsics {
        &self.camera
    }

    /// Diameter of the landmarks' axis-aligned bounding box.
    pub fn extent(&self) -> f64 {
        if self.landmarks.is_empty() {
            return 0.0;
        }
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for landmark in &self.landmarks {
            let c = landmark.position.coords;
            for k in 0..3 {
                min[k] = min[k].min(c[k]);
                max[k] = max[k].max(c[k]);
            }
        }
        ((max[0] - min[0]).powi(2) + (max[1] - min[1]).powi(2) + (max[2] - min[2]).powi(2)).sqrt()
    }

    /// Index (into [`SyntheticScene::views`]) of the database image that shares
    /// the most landmarks with the query — the "correct" retrieval answer.
    pub fn correct_view_index(&self) -> usize {
        self.correct_view
    }

    /// Landmark indices visible in each view, parallel to [`SyntheticScene::views`].
    pub fn view_landmark_ids(&self) -> &[Vec<usize>] {
        &self.view_landmark_ids
    }

    /// Id of the [`SyntheticScene::correct_view_index`] image.
    pub fn correct_view_id(&self) -> usize {
        self.views[self.correct_view].id
    }

    /// Ground-truth pose of the query view.
    pub fn ground_truth_pose(&self) -> Pose {
        self.query.pose
    }

    /// Build a [`Database`] from this scene (cloning the landmarks and views).
    ///
    /// Passing a [`Vocabulary`] additionally enables BoW retrieval.
    pub fn database(&self, vocabulary: Option<Vocabulary>) -> Database {
        let mut database = Database::new(vocabulary);
        for landmark in &self.landmarks {
            database.add_landmark(landmark.clone());
        }
        for image in &self.views {
            database.add_image(image.clone());
        }
        database.build();
        database
    }

    /// A copy of the query view whose keypoint coordinates carry independent
    /// zero-mean Gaussian pixel noise with standard deviation `sigma_px`.
    ///
    /// Descriptors are left untouched (only the *observations* are corrupted),
    /// and `seed` makes the noise reproducible.
    pub fn query_with_pixel_noise(&self, sigma_px: f64, seed: u64) -> Query {
        let mut rng = Rng::new(seed);
        let keypoints = self
            .query
            .keypoints
            .iter()
            .map(|kp| {
                let mut noisy = *kp;
                noisy.x += sigma_px * rng.normal();
                noisy.y += sigma_px * rng.normal();
                noisy
            })
            .collect();

        Query {
            keypoints,
            descriptors: self.query.descriptors.clone(),
            pose: self.query.pose,
            landmark_ids: self.query.landmark_ids.clone(),
        }
    }

    /// A query that shares no landmarks with the database: random descriptors
    /// and random pixel coordinates, so it cannot match anything.
    pub fn no_overlap_query(&self, seed: u64) -> Query {
        let mut rng = Rng::new(seed ^ 0xA5A5_5A5A_A5A5_5A5A);
        let n = self.query.keypoints.len().max(30);
        let mut keypoints = Vec::with_capacity(n);
        let mut descriptors = Descriptors::new();

        for _ in 0..n {
            let x = rng.range(5.0, self.camera.width as f64 - 5.0);
            let y = rng.range(5.0, self.camera.height as f64 - 5.0);
            let kp = KeyPoint::new(x, y);
            keypoints.push(kp);
            descriptors.push(Descriptor::new(rng.bytes(self.descriptor_dim), kp));
        }

        Query {
            keypoints,
            descriptors,
            pose: Pose::identity(),
            landmark_ids: Vec::new(),
        }
    }
}

/// Generate a scene from `config`.
pub fn generate(config: &SyntheticConfig) -> SyntheticScene {
    let mut rng = Rng::new(config.seed);

    let camera = CameraIntrinsics::new(
        config.focal,
        config.focal,
        config.image_width as f64 / 2.0,
        config.image_height as f64 / 2.0,
        config.image_width,
        config.image_height,
    );

    // ---- Landmarks: a random cloud with a unique descriptor each ----
    let half = config.extent / 2.0;
    let mut positions = Vec::with_capacity(config.num_landmarks);
    let mut landmark_descriptors = Vec::with_capacity(config.num_landmarks);
    for _ in 0..config.num_landmarks {
        positions.push(Point3::new(
            rng.range(-half, half),
            rng.range(-half, half),
            rng.range(-half, half),
        ));
        landmark_descriptors.push(rng.bytes(config.descriptor_dim));
    }
    let landmarks: Vec<Landmark> = positions
        .iter()
        .zip(landmark_descriptors.iter())
        .map(|(position, bytes)| Landmark {
            position: *position,
            descriptors: vec![Descriptor::new(bytes.clone(), KeyPoint::new(0.0, 0.0))],
        })
        .collect();

    // ---- Database views on a ring around the cloud ----
    let spacing = std::f64::consts::TAU / config.num_views as f64;
    let height = config.ring_radius * 0.15;
    let mut views = Vec::with_capacity(config.num_views);
    let mut view_landmark_ids = Vec::with_capacity(config.num_views);

    for v in 0..config.num_views {
        let theta = spacing * v as f64;
        let eye = Point3::new(
            config.ring_radius * theta.cos(),
            height,
            config.ring_radius * theta.sin(),
        );
        let pose = look_at(&eye, &Point3::origin());
        let (keypoints, descriptors, ids) =
            observe(&pose, &camera, &positions, &landmark_descriptors);
        views.push(DatabaseImage {
            id: v,
            pose: Some(pose),
            keypoints,
            descriptors,
            landmarks: ids.iter().map(|&id| Some(id)).collect(),
        });
        view_landmark_ids.push(ids);
    }

    // ---- Query: a nearby viewpoint between two ring views ----
    let anchor = config.num_views / 2;
    let theta_q = spacing * (anchor as f64 + 0.35);
    let eye_q = Point3::new(
        config.ring_radius * 0.95 * theta_q.cos(),
        height * 1.1,
        config.ring_radius * 0.95 * theta_q.sin(),
    );
    let query_pose = look_at(&eye_q, &Point3::origin());
    let (keypoints, descriptors, landmark_ids) =
        observe(&query_pose, &camera, &positions, &landmark_descriptors);
    let query = Query {
        keypoints,
        descriptors,
        pose: query_pose,
        landmark_ids,
    };

    // ---- Ground-truth retrieval answer: max landmark overlap ----
    let correct_view = view_landmark_ids
        .iter()
        .enumerate()
        .map(|(v, ids)| {
            let overlap = query
                .landmark_ids
                .iter()
                .filter(|id| ids.contains(id))
                .count();
            (v, overlap)
        })
        .max_by_key(|&(v, overlap)| (overlap, std::cmp::Reverse(v)))
        .map(|(v, _)| v)
        .unwrap_or(0);

    SyntheticScene {
        camera,
        landmarks,
        views,
        view_landmark_ids,
        query,
        correct_view,
        descriptor_dim: config.descriptor_dim,
    }
}

/// Project every landmark into `pose`, keeping those inside the image.
fn observe(
    pose: &Pose,
    camera: &CameraIntrinsics,
    positions: &[Point3<f64>],
    descriptors: &[Vec<u8>],
) -> (Vec<KeyPoint>, Descriptors, Vec<usize>) {
    let margin = 2.0;
    let mut keypoints = Vec::new();
    let mut descs = Descriptors::new();
    let mut ids = Vec::new();

    for (i, position) in positions.iter().enumerate() {
        let camera_point = pose.rotation * position.coords + pose.translation;
        if camera_point[2] <= 1e-6 {
            continue;
        }
        let projected = camera.project(&Point3::from(camera_point));
        if projected.x < margin
            || projected.y < margin
            || projected.x > camera.width as f64 - margin
            || projected.y > camera.height as f64 - margin
        {
            continue;
        }

        let kp = KeyPoint::new(projected.x, projected.y);
        keypoints.push(kp);
        descs.push(Descriptor::new(descriptors[i].clone(), kp));
        ids.push(i);
    }

    (keypoints, descs, ids)
}

/// World-to-camera pose of a camera at `eye` looking at the origin, +Y up.
///
/// Builds a right-handed orthonormal basis with the camera looking along `+z`
/// (the convention used by [`cv_core::CameraIntrinsics::project`]).
fn look_at(eye: &Point3<f64>, target: &Point3<f64>) -> Pose {
    let forward = (target - eye).normalize();
    let right = Vector3::y().cross(&forward).normalize();
    let up = forward.cross(&right);

    let rotation = Matrix3::from_rows(&[right.transpose(), up.transpose(), forward.transpose()]);
    let translation = -(rotation * eye.coords);
    Pose::from_quat_translation(
        UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(rotation)),
        translation,
    )
}

/// Minimal deterministic xorshift64* generator — keeps the crate dependency-free.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform sample in `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform sample in `[lo, hi)`.
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    /// Standard-normal sample via the Box-Muller transform.
    fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// `n` random bytes.
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| (self.next_u64() & 0xFF) as u8).collect()
    }
}
