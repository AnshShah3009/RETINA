//! Is `cv_3d`'s k-NN exact? Compared against brute force on a **tie-free** scene,
//! which is what the lattice-based harness could not settle (docs/PARITY.md,
//! "k-NN neighbour sets": "on a scene with no ties it should match brute force
//! exactly, and that has not been verified").
use cv_3d::spatial::KDTree;
use nalgebra::Point3;

fn main() {
    // Fixed-seed LCG: reproducible, and the small jitter breaks the exact ties a
    // cubic lattice is made of. Distances here are distinct with overwhelming
    // probability, so the k-th boundary is never ambiguous and the neighbour
    // *set* is fully determined.
    let mut s: u64 = 0x5EED_1234_ABCD_0001;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    };
    let n = 200;
    let pts: Vec<Point3<f32>> = (0..n)
        .map(|_| {
            Point3::new(
                next() as f32 * 2.0,
                next() as f32 * 2.0,
                next() as f32 * 2.0,
            )
        })
        .collect();

    for k in [4usize, 6, 10] {
        let mut items: Vec<(Point3<f32>, usize)> = pts
            .iter()
            .copied()
            .enumerate()
            .map(|(j, p)| (p, j))
            .collect();
        let tree = KDTree::build(&mut items);
        let mut exact_set = 0usize;
        for i in 0..n {
            // Brute-force truth: distances computed in f64 (the f32 coordinates
            // promote exactly, then subtraction and norm run in f64), so the
            // ranking is the real Euclidean one.
            let full: Vec<f64> = pts
                .iter()
                .map(|q| {
                    let d = q - pts[i];
                    (d.x as f64 * d.x as f64 + d.y as f64 * d.y as f64 + d.z as f64 * d.z as f64)
                        .sqrt()
                })
                .collect();
            let mut ord: Vec<usize> = (0..n).collect();
            ord.sort_by(|&a, &b| full[a].partial_cmp(&full[b]).unwrap().then(a.cmp(&b)));
            // A tied k-th boundary would leave the neighbour set undetermined,
            // so the scene's tie-free claim is asserted here, not assumed.
            assert!(
                full[ord[k - 1]] < full[ord[k]],
                "k-th boundary tied for query {i}, k={k}"
            );
            let truth: std::collections::BTreeSet<usize> = ord.iter().take(k).copied().collect();
            let got = tree.k_nearest_neighbors(&pts[i], k);
            let set: std::collections::BTreeSet<usize> =
                got.into_iter().map(|(_, j, _)| j).collect();
            if set == truth {
                exact_set += 1;
            }
        }
        println!("k={k}: exact set on {exact_set}/{n} tie-free queries");
    }
}
