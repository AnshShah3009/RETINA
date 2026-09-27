// Temporary diagnostic: dump ORB keypoints/descriptors for a TUM view at a
// given feature budget so the mapper's top-N cut can be inspected.
use cv_features::orb::orb_detect_and_compute;
use std::collections::HashSet;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let feature_counts: Vec<usize> = args[2..].iter().map(|s| s.parse().unwrap()).collect();
    let img = image::open(path).unwrap().to_luma8();
    for n in feature_counts {
        let (kps, descs) = orb_detect_and_compute(&img, n);
        let d: HashSet<(u32, u32)> = descs
            .iter()
            .map(|d| (d.keypoint.x as u32, d.keypoint.y as u32))
            .collect();
        // response histogram of the top-N kept descriptors
        let mut resp: Vec<u64> = descs.iter().map(|d| d.keypoint.response as u64).collect();
        resp.sort_unstable();
        let distinct = {
            let mut v = resp.clone();
            v.dedup();
            v.len()
        };
        println!(
            "N={n:5} keypoints={:5} descriptors={:5} distinct_positions={:5} distinct_responses={:5} \
             resp[0]={} resp[mid]={} resp[last]={}",
            kps.keypoints.len(),
            descs.len(),
            d.len(),
            distinct,
            resp.first().copied().unwrap_or(0),
            resp.get(resp.len() / 2).copied().unwrap_or(0),
            resp.last().copied().unwrap_or(0),
        );
    }
}
