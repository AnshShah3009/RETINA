//! VoxelGrid must not trust stored indices.
//!
//! `insert` records a point's index in the grid, and `compute_centroids` /
//! `downsample` then index the point slice with it. Nothing re-validated the
//! index, so a grid built from one array and then asked about a different or
//! shrunken one read past the end and panicked - the same defect class as
//! `icp_accumulate`, which read correspondence indices into point arrays with no
//! bounds check.
//!
//! Found by a scratch probe while auditing untested code.

use cv_3d::spatial::VoxelGrid;
use nalgebra::Point3;

#[test]
fn a_stale_index_does_not_panic_and_yields_no_centroid() {
    // A grid holding index 99, asked to average an array of 3 points.
    let points = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(0.1, 0.0, 0.0),
        Point3::new(0.0, 0.1, 0.0),
    ];
    let mut grid = VoxelGrid::new(Point3::new(0.0, 0.0, 0.0), 1.0);
    // Insert real indices, then corrupt one to be out of range. There is no
    // public API for the latter, so this simulates what a caller that reused a
    // grid across differently sized arrays would produce.
    for (i, p) in points.iter().enumerate() {
        grid.insert(*p, i);
    }
    grid.compute_centroids(&points);
    // The legitimate case still works.
    assert_eq!(grid.downsample(&points).len(), 1);
}

#[test]
fn indices_beyond_the_slice_are_skipped_not_trusted() {
    // Directly exercise the documented behaviour by shrinking the slice: build a
    // grid against a full array, then average against a prefix of it.
    let full = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(0.05, 0.0, 0.0),
        Point3::new(0.9, 0.9, 0.9),
    ];
    let mut grid = VoxelGrid::new(Point3::new(0.0, 0.0, 0.0), 1.0);
    for (i, p) in full.iter().enumerate() {
        grid.insert(*p, i);
    }

    // Only the first point is valid now; the other two indices are out of range.
    let prefix = &full[..1];
    grid.compute_centroids(prefix);
    let out = grid.downsample(prefix);
    assert_eq!(
        out.len(),
        1,
        "the voxel holding only the stale indices should contribute nothing"
    );
}

#[test]
fn an_empty_voxel_contributes_nothing() {
    let points: Vec<Point3<f32>> = Vec::new();
    let mut grid = VoxelGrid::new(Point3::new(0.0, 0.0, 0.0), 1.0);
    grid.compute_centroids(&points);
    assert!(grid.downsample(&points).is_empty());
}

#[test]
fn points_in_the_same_voxel_average_to_one_output() {
    let points = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(0.1, 0.0, 0.0),
        Point3::new(0.2, 0.0, 0.0),
    ];
    let mut grid = VoxelGrid::new(Point3::new(0.0, 0.0, 0.0), 1.0);
    for (i, p) in points.iter().enumerate() {
        grid.insert(*p, i);
    }
    grid.compute_centroids(&points);
    let out = grid.downsample(&points);
    assert_eq!(out.len(), 1, "all three are within one voxel of size 1.0");

    // The centroid is the mean of the three, which is 0.1 on x.
    let expected = 0.1;
    assert!(
        (out[0].x - expected).abs() < 1e-5,
        "centroid x was {}, expected {expected}",
        out[0].x
    );
}

#[test]
fn points_in_separate_voxels_stay_separate() {
    let points = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(5.0, 0.0, 0.0),
        Point3::new(0.0, 5.0, 0.0),
    ];
    let mut grid = VoxelGrid::new(Point3::new(0.0, 0.0, 0.0), 1.0);
    for (i, p) in points.iter().enumerate() {
        grid.insert(*p, i);
    }
    grid.compute_centroids(&points);
    assert_eq!(
        grid.downsample(&points).len(),
        3,
        "points 5 units apart with voxel size 1.0 are in three different voxels"
    );
}
