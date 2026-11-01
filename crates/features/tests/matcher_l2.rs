//! `MatchType::BruteForceL2` matches gradient-histogram descriptors by
//! Euclidean distance. With Hamming, query byte 128 is 8 bits from 127 but 1
//! bit from 0, so the far descriptor won.

use cv_core::{Descriptor, KeyPoint};
use cv_features::descriptor::Descriptors;
use cv_features::matcher::{MatchType, Matcher};

fn descs(rows: &[[u8; 4]]) -> Descriptors {
    let mut d = Descriptors::new();
    for r in rows {
        d.push(Descriptor::new(r.to_vec(), KeyPoint::new(0.0, 0.0)));
    }
    d
}

#[test]
fn l2_picks_the_numerically_nearest_descriptor() {
    let query = descs(&[[128, 10, 10, 10]]);
    let train = descs(&[[0, 10, 10, 10], [127, 10, 10, 10]]);
    let l2 = Matcher::new(MatchType::BruteForceL2).match_descriptors(&query, &train);
    assert_eq!(l2.matches[0].train_idx, 1);
    assert!((l2.matches[0].distance - 1.0).abs() < 1e-6);
    // Hamming picks the other one - which is why SIFT needs L2.
    let ham = Matcher::new(MatchType::BruteForceHamming).match_descriptors(&query, &train);
    assert_eq!(ham.matches[0].train_idx, 0);
}

fn descs_raw(rows: &[&[u8]]) -> Descriptors {
    let mut d = Descriptors::new();
    for r in rows {
        d.push(Descriptor::new(r.to_vec(), KeyPoint::new(0.0, 0.0)));
    }
    d
}

#[test]
fn cross_check_against_an_empty_train_set_does_not_panic() {
    let query = descs(&[[1, 2, 3, 4]]);
    let train = descs(&[]);
    let m = Matcher::new(MatchType::BruteForceL2)
        .with_cross_check()
        .match_descriptors(&query, &train);
    assert!(m.matches.is_empty());
}

/// An 8-byte zero descriptor against a 128-byte zero descriptor used to
/// report L2 distance 0, because `zip` stopped at the shorter slice.
#[test]
fn l2_short_zeros_against_long_zeros_is_not_distance_zero() {
    let query = descs_raw(&[&[0u8; 8]]);
    let train = descs_raw(&[&[0u8; 128]]);
    let m = Matcher::new(MatchType::BruteForceL2).match_descriptors(&query, &train);
    assert_eq!(m.matches.len(), 1);
    assert_eq!(
        m.matches[0].distance,
        u32::MAX as f32,
        "unequal lengths must take the Hamming sentinel, not a prefix distance"
    );
}

/// A longer descriptor that only shares a zero prefix must not beat a real
/// equal-length neighbor. The prefix used to look identical under `zip`.
#[test]
fn l2_prefix_lookalike_does_not_beat_equal_length_neighbor() {
    let query = descs_raw(&[&[0u8; 8]]);
    let mut lookalike = vec![0u8; 128];
    lookalike[8..].fill(255);
    let train = descs_raw(&[&lookalike, &[0u8; 8]]);
    let m = Matcher::new(MatchType::BruteForceL2).match_descriptors(&query, &train);
    assert_eq!(m.matches.len(), 1);
    assert_eq!(m.matches[0].train_idx, 1);
    assert!(m.matches[0].distance.abs() < 1e-6);
}
