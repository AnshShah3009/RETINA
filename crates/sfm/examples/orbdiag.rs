// Temporary diagnostic: lift ORB matches of a contiguous view window to
// keypoint indices and feed them to the mapper, to check whether a change in
// the feature budget silently re-pairs keypoints with the wrong descriptors.
use cv_core::CameraIntrinsics;
use cv_features::matcher::{MatchType, Matcher};
use cv_features::orb::orb_detect_and_compute;
use cv_sfm::mapper::{map_views, MapperConfig, PairSelection, View};

type Pos = (i64, i64);

fn pos(k: &cv_core::KeyPoint) -> Pos {
    (k.x as i64, k.y as i64)
}

fn load(dir: &str, n: usize) -> Vec<(String, image::GrayImage)> {
    let text = std::fs::read_to_string(format!("{dir}/rgb.txt")).unwrap();
    text.lines()
        .skip(3)
        .filter(|l| !l.trim().is_empty())
        .take(n)
        .map(|l| {
            let f = l.split_whitespace().nth(1).unwrap().to_string();
            let img = image::open(format!("{dir}/{f}")).unwrap().to_luma8();
            (f, img)
        })
        .collect()
}

fn main() {
    let dir = "/home/Phoenix/RUST/datasets/rgbd_dataset_freiburg1_desk";
    let frames: Vec<(String, image::GrayImage)> = load(dir, 20);
    let intrinsics = CameraIntrinsics::new(517.3, 516.5, 318.6, 255.3, 640, 480);

    for n in [800usize, 1000, 1200, 1400, 1500, 1600, 2000, 3000] {
        let mut views: Vec<View> = Vec::new();
        let mut n_desc = 0usize;
        let mut n_distinct = 0usize;
        for (_name, img) in &frames {
            let (_kps, descs) = orb_detect_and_compute(img, n);
            let keypoints: Vec<_> = descs.iter().map(|d| d.keypoint).collect();
            n_desc += keypoints.len();
            n_distinct += keypoints
                .iter()
                .map(pos)
                .collect::<std::collections::HashSet<_>>()
                .len();
            views.push(View::new(keypoints, descs, img.width(), img.height()));
        }

        // (a) direct check: does descriptor i live at keypoint i?
        let geom_aligned = views.iter().all(|v| {
            (0..v.len())
                .all(|i| pos(&v.descriptors.descriptors[i].keypoint) == pos(&v.keypoints[i]))
        });

        // (b) does a known strong feature keep its keypoint INDEX as the budget
        // changes? (The mapper's keypoint indices are per-view, so only
        // intra-view stability matters, but the same instability is what makes
        // tracks jump between runs.)
        let anchor_view = &views[10];
        let anchor_distinct: std::collections::HashSet<Pos> =
            anchor_view.keypoints.iter().map(pos).collect();
        let anchor_at = |x: i64, y: i64| -> Vec<usize> {
            (0..anchor_view.len())
                .filter(|&i| pos(&anchor_view.keypoints[i]) == (x, y))
                .collect()
        };

        // (c) how many verified-pair inliers are on duplicate keypoints?
        let matcher = Matcher::new(MatchType::BruteForce)
            .with_ratio_test(0.75)
            .with_cross_check();
        let mut dup_inliers = 0usize;
        let mut total_inliers = 0usize;
        let mut verified = 0usize;
        for w in 0..4 {
            let a = w * 5;
            let b = a + 1;
            let m = matcher.match_descriptors(&views[a].descriptors, &views[b].descriptors);
            if m.matches.len() < 20 {
                continue;
            }
            let pa: Vec<_> = m
                .matches
                .iter()
                .map(|x| views[a].keypoints[x.query_idx as usize].pt())
                .collect();
            let pb: Vec<_> = m
                .matches
                .iter()
                .map(|x| views[b].keypoints[x.train_idx as usize].pt())
                .collect();
            if let Ok(_f) = cv_calib3d::find_fundamental_mat(&pa, &pb) {
                verified += 1;
                for (i, &is_inlier) in vec![true; m.matches.len()].iter().enumerate() {
                    if is_inlier {
                        total_inliers += 1;
                        let q = m.matches[i].query_idx as usize;
                        let t = m.matches[i].train_idx as usize;
                        let qd = views[a]
                            .keypoints
                            .iter()
                            .filter(|k| pos(k) == pos(&views[a].keypoints[q]))
                            .count();
                        let td = views[b]
                            .keypoints
                            .iter()
                            .filter(|k| pos(k) == pos(&views[b].keypoints[t]))
                            .count();
                        if qd > 1 || td > 1 {
                            dup_inliers += 1;
                        }
                    }
                }
            }
        }

        // (d) end-to-end mapper on the same 20 views.
        let config = MapperConfig {
            pair_selection: PairSelection::Sequential { window: 5 },
            seed_hypotheses: 8,
            ..MapperConfig::default()
        };
        let mapping = map_views(&views, &intrinsics, &config);

        println!(
            "N={n:5} geom_aligned={geom_aligned} descriptors={n_desc} distinct={n_distinct} \
             dup_positions={} anchor_320,240_idx={:?} verified_pairs={verified} inliers={total_inliers} \
             inliers_on_dup_keypoints={dup_inliers} => registered={}/{} seed={:?} points={}",
            n_desc - n_distinct,
            anchor_at(320, 240),
            mapping.report.registered,
            mapping.report.views_supplied,
            mapping.report.seed,
            mapping.report.num_points
        );
        let _ = anchor_distinct;
    }
}
