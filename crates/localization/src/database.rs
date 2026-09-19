//! The landmark database that a [`crate::Localizer`] queries against.

use cv_core::{Descriptor, Descriptors, KeyPoint, Pose};
use cv_features::retrieval::{BowDatabase, LshIndex, Vocabulary};
use nalgebra::Point3;
use std::collections::HashMap;

/// A 3D point together with the descriptors of the keypoints that observed it.
///
/// The `descriptors` are the *observations* of this landmark: typically the same
/// visual word seen from several viewpoints, stored as one
/// [`Descriptor`](cv_core::Descriptor) per observing keypoint.
#[derive(Debug, Clone)]
pub struct Landmark {
    /// Position of the point in the world frame.
    pub position: Point3<f64>,
    /// Descriptors of the keypoints that observed this landmark.
    pub descriptors: Vec<Descriptor>,
}

/// One image of the database.
///
/// `landmarks[i]` is the index (into [`Database::landmarks`]) of the 3D point
/// observed by keypoint `i`, or `None` for keypoints that were not triangulated.
#[derive(Debug, Clone)]
pub struct DatabaseImage {
    /// Caller-chosen image identifier. Unique within a database.
    pub id: usize,
    /// World-to-camera pose (`x_cam = R * X_world + t`), if the image is
    /// registered. Localization only needs the *candidate* pose for bookkeeping;
    /// the query pose is recovered from the landmarks.
    pub pose: Option<Pose>,
    /// Detected keypoints, parallel to `descriptors` and `landmarks`.
    pub keypoints: Vec<KeyPoint>,
    /// Descriptors, parallel to `keypoints`.
    pub descriptors: Descriptors,
    /// Landmark index for each keypoint (`None` when not triangulated).
    pub landmarks: Vec<Option<usize>>,
}

/// Maps a global database-descriptor index to the image/keypoint it came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DescriptorRef {
    /// Identifier of the image the descriptor belongs to.
    pub image_id: usize,
    /// Index of the descriptor within that image.
    pub keypoint: usize,
}

/// A collection of [`DatabaseImage`]s and [`Landmark`]s that can localize a query.
///
/// Add images and landmarks, then call [`Database::build`] once to construct the
/// retrieval indices. Building is idempotent and safe on an empty database.
pub struct Database {
    vocabulary: Option<Vocabulary>,
    images: Vec<DatabaseImage>,
    landmarks: Vec<Landmark>,
    id_to_index: HashMap<usize, usize>,
    /// Flat concatenation of every image's descriptors, in image/keypoint order.
    descriptors: Descriptors,
    /// Parallel to `descriptors`: origin of each flat descriptor.
    descriptor_index: Vec<DescriptorRef>,
    /// Inverted-file BoW index, present only when a vocabulary was supplied.
    bow: Option<BowDatabase>,
    /// Approximate nearest-neighbour index over the flat descriptors.
    lsh: Option<LshIndex>,
    built: bool,
}

impl Database {
    /// Create an empty database.
    ///
    /// When `vocabulary` is `Some`, [`Database::build`] also constructs a BoW
    /// inverted index used for candidate retrieval.
    pub fn new(vocabulary: Option<Vocabulary>) -> Self {
        Self {
            vocabulary,
            images: Vec::new(),
            landmarks: Vec::new(),
            id_to_index: HashMap::new(),
            descriptors: Descriptors::new(),
            descriptor_index: Vec::new(),
            bow: None,
            lsh: None,
            built: false,
        }
    }

    /// Add a database image. Any previously built index is invalidated until
    /// [`Database::build`] is called again.
    pub fn add_image(&mut self, image: DatabaseImage) {
        self.id_to_index.insert(image.id, self.images.len());
        self.images.push(image);
        self.built = false;
    }

    /// Add a landmark. Landmarks are addressed by their insertion index.
    pub fn add_landmark(&mut self, landmark: Landmark) {
        self.landmarks.push(landmark);
        self.built = false;
    }

    /// Build the retrieval indices: a flat descriptor table, an LSH index over
    /// it, and (when a vocabulary is present) a BoW inverted index.
    pub fn build(&mut self) {
        let mut flat = Descriptors::new();
        let mut index = Vec::new();
        for image in &self.images {
            for (keypoint, descriptor) in image.descriptors.iter().enumerate() {
                flat.push(descriptor.clone());
                index.push(DescriptorRef {
                    image_id: image.id,
                    keypoint,
                });
            }
        }

        let dim = flat.iter().map(|d| d.data.len()).max().unwrap_or(0);
        self.lsh = if dim == 0 || flat.is_empty() {
            None
        } else {
            // Fixed layout/seed keeps the index (and therefore query results)
            // fully deterministic.
            let mut lsh = LshIndex::new(dim, 8, 16, 0x5EED_1DEA);
            for (id, descriptor) in flat.iter().enumerate() {
                lsh.insert(id, descriptor.data.as_slice());
            }
            Some(lsh)
        };

        self.bow = self.vocabulary.clone().map(|vocabulary| {
            let mut bow = BowDatabase::new(vocabulary);
            for image in &self.images {
                let bytes: Vec<&[u8]> = image
                    .descriptors
                    .iter()
                    .map(|d| d.data.as_slice())
                    .collect();
                bow.add(image.id, &bytes);
            }
            bow.build();
            bow
        });

        self.descriptors = flat;
        self.descriptor_index = index;
        self.built = true;
    }

    /// Number of database images.
    pub fn len(&self) -> usize {
        self.images.len()
    }

    /// Whether the database holds no images.
    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    /// Whether [`Database::build`] has run since the last modification.
    pub fn is_built(&self) -> bool {
        self.built
    }

    /// Image identifiers in insertion order.
    pub fn image_ids(&self) -> Vec<usize> {
        self.images.iter().map(|image| image.id).collect()
    }

    /// The flat descriptor table and its per-entry origin, parallel to each other.
    ///
    /// Empty until [`Database::build`] has run.
    pub fn descriptor_index(&self) -> &[DescriptorRef] {
        &self.descriptor_index
    }

    /// The flat concatenation of every image's descriptors.
    pub fn descriptors(&self) -> &Descriptors {
        &self.descriptors
    }

    /// Number of landmarks.
    pub fn landmark_count(&self) -> usize {
        self.landmarks.len()
    }

    /// The landmark at `index`, if any.
    pub fn landmark(&self, index: usize) -> Option<&Landmark> {
        self.landmarks.get(index)
    }

    /// Every landmark, in insertion order.
    pub fn landmarks(&self) -> &[Landmark] {
        &self.landmarks
    }

    /// Every image, in insertion order.
    pub fn images(&self) -> &[DatabaseImage] {
        &self.images
    }

    /// The image with the given `id`, if present.
    pub fn image(&self, id: usize) -> Option<&DatabaseImage> {
        self.id_to_index.get(&id).and_then(|&i| self.images.get(i))
    }

    /// The vocabulary backing retrieval, if one was supplied.
    pub fn vocabulary(&self) -> Option<&Vocabulary> {
        self.vocabulary.as_ref()
    }

    /// The BoW inverted index, present only after [`Database::build`] with a vocabulary.
    pub fn bow_index(&self) -> Option<&BowDatabase> {
        self.bow.as_ref()
    }

    /// The LSH index over the flat descriptors, present only after a build with
    /// at least one non-empty descriptor.
    pub fn lsh_index(&self) -> Option<&LshIndex> {
        self.lsh.as_ref()
    }

    /// Rank images by how many query descriptors each one can match.
    ///
    /// For every image the count is the number of query descriptors whose
    /// nearest descriptor *within that image* lies within a Hamming threshold of
    /// `25%` of the descriptor length. Images with a non-zero count are returned
    /// best-first with ties broken by ascending image id, truncated to `k`.
    ///
    /// This is the fallback retrieval path used when no vocabulary is available.
    /// It returns an empty vector for an empty query, an empty database or `k == 0`.
    pub fn rank_by_descriptor_matches(&self, query: &Descriptors, k: usize) -> Vec<usize> {
        if k == 0 || query.is_empty() || self.images.is_empty() {
            return Vec::new();
        }

        let dim_bytes = self
            .images
            .iter()
            .flat_map(|image| image.descriptors.iter())
            .map(|d| d.data.len())
            .max()
            .unwrap_or(0);
        let threshold = ((dim_bytes * 8) / 4) as u32;

        let mut ranked: Vec<(usize, usize)> = self
            .images
            .iter()
            .map(|image| {
                let count = query
                    .iter()
                    .filter(|qd| {
                        image
                            .descriptors
                            .iter()
                            .map(|d| qd.hamming_distance(d))
                            .min()
                            .map(|dist| dist <= threshold)
                            .unwrap_or(false)
                    })
                    .count();
                (image.id, count)
            })
            .collect();

        ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked.retain(|&(_, count)| count > 0);
        ranked.truncate(k);
        ranked.into_iter().map(|(id, _)| id).collect()
    }
}

/// Build a [`DatabaseImage`] skeleton from a COLMAP text-model image.
///
/// COLMAP's text model stores 2D observations and their `POINT3D_ID`s but no
/// descriptors, so the returned image has empty [`DatabaseImage::descriptors`];
/// callers fill them from the pixels. `landmark_of` maps a COLMAP
/// `POINT3D_ID` to this database's landmark index; observations with id `-1`
/// (un-triangulated) or ids missing from the map become `None`.
pub fn database_image_from_colmap(
    image: &cv_io::datasets::colmap::Image,
    landmark_of: &HashMap<u64, usize>,
) -> DatabaseImage {
    let keypoints: Vec<KeyPoint> = image
        .points2d
        .iter()
        .map(|p| KeyPoint::new(p.x, p.y))
        .collect();
    let landmarks: Vec<Option<usize>> = image
        .points2d
        .iter()
        .map(|p| {
            if p.point3d_id < 0 {
                None
            } else {
                landmark_of.get(&(p.point3d_id as u64)).copied()
            }
        })
        .collect();

    DatabaseImage {
        id: image.id as usize,
        pose: Some(image.pose),
        keypoints,
        descriptors: Descriptors::new(),
        landmarks,
    }
}
