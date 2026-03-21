use cv_scientific::fft::{fft, ifft};
use cv_scientific::spatial::KDTree;
use std::time::Instant;

#[test]
fn profile_scientific_algorithms() {
    println!("--- Profiling cv-scientific ---");
    let num_elements = 1_000_000;

    // 1. FFT
    let data: Vec<f64> = (0..num_elements).map(|x| (x as f64).sin()).collect();
    let start = Instant::now();
    let complex_data = fft(&data);
    let elapsed = start.elapsed();
    println!("FFT (1M elements): {:?}", elapsed);
    assert_eq!(complex_data.len(), num_elements);

    // 2. IFFT
    let start = Instant::now();
    let _ = ifft(&complex_data);
    let elapsed = start.elapsed();
    println!("IFFT (1M elements): {:?}", elapsed);

    // 3. KDTree
    let num_points = 100_000;
    let points: Vec<Vec<f64>> = (0..num_points)
        .map(|i| vec![i as f64 * 0.1, i as f64 * 0.2, i as f64 * 0.3])
        .collect();

    let start = Instant::now();
    let tree = KDTree::new(&points).unwrap();
    let elapsed_build = start.elapsed();
    println!("KDTree Build (100k points, 3D): {:?}", elapsed_build);

    let query = vec![50.0, 100.0, 150.0];
    let start = Instant::now();
    let _ = tree.query(&query, 10);
    let elapsed_query = start.elapsed();
    println!("KDTree Query (k=10): {:?}", elapsed_query);
}
