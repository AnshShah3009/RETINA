//! `search_radius` accepts points at exactly `radius` (`dist <= r²`), so the
//! subtree prune must be inclusive too. With a strict `<` the far subtree was
//! pruned when the splitting plane sat exactly at the radius, and duplicates
//! of a boundary point came back 2 of 3.

use cv_3d::KDTree;
use nalgebra::Point3;

#[test]
fn every_point_on_the_sphere_is_returned() {
    let mut tree = KDTree::new();
    for i in 0..3 {
        tree.insert(Point3::new(1.0, 0.0, 0.0), i);
    }
    let hits = tree.search_radius(&Point3::origin(), 1.0);
    assert_eq!(hits.len(), 3, "{hits:?}");
}
