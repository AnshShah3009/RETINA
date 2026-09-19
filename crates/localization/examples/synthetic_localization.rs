//! Runnable end-to-end demonstration of the localization pipeline.
//!
//! Enable with `--features synthetic`:
//!
//! ```text
//! cargo run -p cv-localization --features synthetic --example synthetic_localization
//! ```

use cv_localization::synthetic::{generate, SyntheticConfig};
use cv_localization::{evaluate_localization, Localizer, LocalizerConfig, Vocabulary};

fn main() {
    let scene = generate(&SyntheticConfig::default());

    // Build a database with BoW retrieval (a vocabulary over the landmarks).
    let pool: Vec<Vec<u8>> = scene
        .landmarks
        .iter()
        .flat_map(|landmark| landmark.descriptors.iter().map(|d| d.data.clone()))
        .collect();
    let vocabulary = Vocabulary::train(&pool, 64, 8, 7);
    let database = scene.database(Some(vocabulary));

    println!(
        "scene: {} landmarks, {} views, extent {:.2}",
        scene.landmarks.len(),
        scene.views.len(),
        scene.extent()
    );

    let localizer = Localizer::new(&database, LocalizerConfig::default());
    let result = localizer.localize(
        &scene.query.keypoints,
        &scene.query.descriptors,
        &scene.camera,
    );

    match result {
        Some(result) => {
            let stats =
                evaluate_localization(&[Some(result.clone())], &[scene.ground_truth_pose()]);
            println!(
                "localized: inliers={} matches={} candidates={:?} rmse={:.3}px",
                result.inliers, result.matches, result.candidates, result.reprojection_rmse
            );
            println!(
                "translation error {:.3e}, rotation error {:.3e} deg, success rate {:.2}",
                stats.mean_translation_error, stats.mean_rotation_error_deg, stats.success_rate
            );
        }
        None => println!("no localization found"),
    }
}
