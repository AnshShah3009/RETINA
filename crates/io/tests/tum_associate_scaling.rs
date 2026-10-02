//! `associate` must stay bit-for-bit compatible with the algorithm it replaced,
//! while no longer building the cross product of the two lists.
//!
//! ## The defect
//!
//! The shipped `associate` collected *every* pair within `max_dt` into a
//! `Vec<(f64, usize, usize)>`, sorted it, and only then consumed the pairs
//! greedily. Both time and memory are `O(n * m)` in the worst case, and `n * m`
//! pairs is reached whenever both lists' timestamps fall inside a single
//! tolerance window. Measured on two lists whose stamps are quantised onto a grid
//! finer than `max_dt` (a plausible `rgb.txt` / `depth.txt` pair over a short
//! window), against the previous implementation:
//!
//! ```text
//!    n     candidate pairs    wall clock    peak RSS
//!  2 000           4 000 000     0.20 s        141 MiB
//! 10 000         100 000 000     7.49 s       3 436 MiB
//! ```
//!
//! At 30 000 that is 900 000 000 pairs and roughly 20 GiB — a machine taken out
//! of memory by two plausible dataset files, through a public function.
//!
//! ## What the replacement may not assume
//!
//! The rule is a **global** greedy: form every candidate pair, sort by
//! `(difference, a index, b index)`, then take a pair whenever neither entry is
//! already used. The replacement here merges runs of consecutive `a` entries that
//! share the same nearest unused `b` entry, and the subtleties below are all cases
//! where the obvious shorter version of that idea disagrees with the oracle. The
//! randomised corpus at the end is built to hit them rather than to look busy.
//!
//! The oracle below is a verbatim copy of the old implementation. It is O(n·m),
//! so every case that runs it is kept small;
//! [`dense_matches_peak_rss_stays_linear`] is the one place a large `n` is used,
//! and it deliberately does *not* call the oracle.

use cv_io::datasets::tum::{associate, IndexEntry};

/// A process-unique suffix, for the temp file this file writes.
///
/// The suite runs tests concurrently inside one process, and a temp file left
/// behind by an aborted earlier run is indistinguishable from one just written,
/// so the name carries both the process id and a per-process counter.
fn unique_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let seq = N.fetch_add(1, Ordering::Relaxed);
    std::process::id() as u64 * 1_000_000 + seq
}

fn entry(timestamp: f64, filename: &str) -> IndexEntry {
    IndexEntry {
        timestamp,
        filename: filename.to_owned(),
    }
}

/// A tiny deterministic PRNG, so a failure is reproducible from the seed in the
/// panic message without pulling in a dependency.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407)
            | 1)
    }

    fn next_u64(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

// ---------------------------------------------------------------------------
// Reference oracle: the previous implementation, verbatim.
// ---------------------------------------------------------------------------

/// The `associate` that shipped before the sweep replaced it, kept here as the
/// specification. O(n·m) time and memory by construction.
fn associate_reference(a: &[IndexEntry], b: &[IndexEntry], max_dt: f64) -> Vec<(usize, usize)> {
    // `max_dt = +inf` is legitimate and means "every pair is a candidate". The
    // oracle originally rejected it, because it shared an `is_finite` test with NaN
    // - so the oracle and the implementation disagreed about a case the test itself
    // then asserted should produce a full pairing.
    if max_dt.is_nan() || max_dt < 0.0 {
        return Vec::new();
    }

    // All candidate pairs within the tolerance, tagged with their difference.
    let mut candidates: Vec<(f64, usize, usize)> = Vec::new();
    for (i, ea) in a.iter().enumerate() {
        for (j, eb) in b.iter().enumerate() {
            let diff = (ea.timestamp - eb.timestamp).abs();
            if diff.is_finite() && diff <= max_dt {
                candidates.push((diff, i, j));
            }
        }
    }

    // Greedy: smallest difference first; indices break ties deterministically.
    candidates.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)).then(x.2.cmp(&y.2)));

    let mut used_a = vec![false; a.len()];
    let mut used_b = vec![false; b.len()];
    let mut matches: Vec<(usize, usize)> = Vec::new();

    for (_, i, j) in candidates {
        if !used_a[i] && !used_b[j] {
            used_a[i] = true;
            used_b[j] = true;
            matches.push((i, j));
        }
    }

    // Match `associate.py`'s final ordering: by the first list's timestamp.
    matches.sort_by(|x, y| {
        a[x.0]
            .timestamp
            .total_cmp(&a[y.0].timestamp)
            .then(x.1.cmp(&y.1))
    });

    matches
}

// ---------------------------------------------------------------------------
// Differential harness
// ---------------------------------------------------------------------------

/// A description of a case, for assertion messages.
struct Case {
    name: &'static str,
    a: Vec<IndexEntry>,
    b: Vec<IndexEntry>,
    max_dt: f64,
    /// Whether an empty result would itself be a bug.
    ///
    /// Not every case should match: `max_dt = inf` on a list whose stamps are
    /// far apart still finds nothing, and `max_dt = 0` on stamps that never
    /// coincide finds nothing. Asserting non-emptiness for those proves nothing
    /// and, worse, *rejects the correct answer* - so the requirement is stated
    /// per case rather than assumed.
    expect_non_empty: bool,
}

impl Case {
    /// A case where a match is expected — the default, and what nearly every case
    /// wants, since an empty result would let a broken implementation pass
    /// silently.
    fn new(name: &'static str, a: Vec<IndexEntry>, b: Vec<IndexEntry>, max_dt: f64) -> Self {
        Case {
            name,
            a,
            b,
            max_dt,
            expect_non_empty: true,
        }
    }

    /// A case where the correct answer may legitimately be empty.
    fn allowing_empty(mut self) -> Self {
        self.expect_non_empty = false;
        self
    }

    fn ts(list: &[IndexEntry]) -> String {
        list.iter()
            .map(|e| format!("{:?}", e.timestamp))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Assert the implementation and the reference agree.
    ///
    /// The control is the third check: a pair is only ever reported when the
    /// reference reports it, the lists here are long enough that an empty result
    /// would be a real failure rather than a vacuous pass, and the checks below
    /// pin the properties the reference's own output is supposed to have. Without
    /// them a "reference" that always returned the same thing would pass
    /// everything.
    fn assert_identical_to_reference(&self) {
        let got = associate(&self.a, &self.b, self.max_dt);
        let want = associate_reference(&self.a, &self.b, self.max_dt);

        assert_eq!(
            got, want,
            "implementation and reference disagree for case {:?}\n  a timestamps: [{}]\n  b timestamps: [{}]\n  max_dt: {:?}",
            self.name,
            Self::ts(&self.a),
            Self::ts(&self.b),
            self.max_dt,
        );

        // Control 1 — the case is not vacuous. A broken implementation that dropped
        // the whole answer would sail through a case whose reference output is
        // empty, so every case that *expects* a match asserts a non-empty result.
        //
        // Cases where emptiness is the correct answer opt out with
        // `allowing_empty()`: `max_dt = inf` across widely separated stamps still
        // finds nothing, and `max_dt = 0` on stamps that never coincide finds
        // nothing. Demanding non-emptiness there would reject a correct answer.
        if self.expect_non_empty {
            assert!(
                !want.is_empty(),
                "case {:?} is vacuous: the reference itself finds no association, \
                 so comparing the two proves nothing",
                self.name,
            );
        }

        // Control 2 — every emitted pair is inside the tolerance, uses each index
        // at most once, and the result really is ordered by (a timestamp, b index).
        for &(i, j) in &got {
            assert!(i < self.a.len() && j < self.b.len(), "index out of range");
            let diff = (self.a[i].timestamp - self.b[j].timestamp).abs();
            assert!(
                diff.is_finite() && diff <= self.max_dt,
                "case {:?}: pair ({i}, {j}) is {diff} apart, beyond max_dt {:?}",
                self.name,
                self.max_dt,
            );
        }
        let seen_a: Vec<usize> = got.iter().map(|p| p.0).collect();
        let seen_b: Vec<usize> = got.iter().map(|p| p.1).collect();
        assert_eq!(
            seen_a.len(),
            seen_a
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            "case {:?}: an a index is used twice",
            self.name
        );
        assert_eq!(
            seen_b.len(),
            seen_b
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            "case {:?}: a b index is used twice",
            self.name
        );
        for &(i, j) in &got {
            assert!(
                self.a[i].timestamp.is_finite(),
                "case {:?}: pair ({i}, {j}) uses a non-finite a timestamp",
                self.name
            );
        }

        // The result must be sorted **lexicographically** by
        // `(a timestamp, b index)`.
        //
        // The previous assertion accepted only `(Less, Equal)` or
        // `(Equal, Less)` - i.e. it required consecutive pairs to have the same
        // `b` index, which no correct implementation can produce, since each
        // `b` index is used at most once. On `a = [0, 10, 20]`,
        // `b = [0.01, 10.5, 19.99]`, `max_dt = 0.1` the correct result
        // `[(0,0), (2,2)]` has timestamps `[0.0, 20.0]` and failed, with the
        // message "not ordered by a's timestamp" - which was itself wrong.
        let keys: Vec<(f64, usize)> = got.iter().map(|&(i, j)| (self.a[i].timestamp, j)).collect();
        for w in keys.windows(2) {
            let ordered = w[0].0.total_cmp(&w[1].0) == std::cmp::Ordering::Less
                || (w[0].0.total_cmp(&w[1].0) == std::cmp::Ordering::Equal && w[0].1 <= w[1].1);
            assert!(
                ordered,
                "case {:?}: result is not sorted by (a timestamp, b index): \
                 {got:?} (a timestamps {:?})",
                self.name,
                keys.iter().map(|k| k.0).collect::<Vec<_>>(),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Explicit edge cases
// ---------------------------------------------------------------------------

/// Both lists empty, and the two one-sided-empty shapes.
#[test]
fn empty_and_one_sided_empty() {
    let x = [entry(0.0, "x")];
    let y = [entry(0.0, "y")];
    for (n, a, b) in [
        ("both empty", &[][..], &[][..]),
        ("a empty", &[][..], &y[..]),
        ("b empty", &x[..], &[][..]),
    ] {
        assert!(
            associate(a, b, 1.0).is_empty(),
            "{n}: an empty list must produce no association"
        );
    }
    // Control: the same lists, with the tolerance widened, do associate — so
    // the emptiness above is the empty list's doing, not a blanket early return.
    assert_eq!(associate(&x, &y, 1.0), vec![(0, 0)]);
}

/// Exact timestamp matches, including several at the same stamp.
#[test]
fn exact_matches() {
    let a = [entry(1.0, "a0"), entry(2.0, "a1"), entry(3.0, "a2")];
    let b = [entry(1.0, "b0"), entry(2.0, "b1"), entry(3.0, "b2")];
    assert_eq!(associate(&a, &b, 0.0), vec![(0, 0), (1, 1), (2, 2)]);
    // Control: a zero tolerance on a *perturbed* list finds nothing, so the
    // result above came from the matches, not from the tolerance.
    let a2 = [entry(1.0, "a0"), entry(2.0, "a1")];
    let b2 = [entry(1.0 + 1e-9, "b0"), entry(2.0 + 1e-9, "b1")];
    assert!(associate(&a2, &b2, 0.0).is_empty());
}

/// The tie case: many entries at identical timestamps, so every candidate pair
/// has the same difference. An implementation that compared differences alone
/// would pick a different pairing than the `(diff, i, j)` sort the reference uses.
#[test]
fn many_pairs_at_identical_timestamps() {
    let a: Vec<IndexEntry> = (0..7).map(|i| entry(0.0, &format!("a{i}"))).collect();
    let b: Vec<IndexEntry> = (0..5).map(|i| entry(0.0, &format!("b{i}"))).collect();
    let case = Case::new("all identical timestamps", a, b, 0.5);
    case.assert_identical_to_reference();

    // Control, spelled out: every difference here is exactly zero, so the whole
    // comparison rests on the index tie-break. Say so explicitly, and pin the
    // shape the tie-break produces.
    assert!(case.a.iter().all(|e| e.timestamp == 0.0));
    assert!(case.b.iter().all(|e| e.timestamp == 0.0));
    let got = associate(&case.a, &case.b, 0.5);
    assert_eq!(
        got,
        vec![(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)],
        "with all differences equal, the lowest unused pair in each list wins"
    );

    // And the two-sided variant: duplicates in a only, then in b only.
    let a2 = vec![entry(5.0, "a0"), entry(5.0, "a1"), entry(1.0, "a2")];
    let b2 = vec![entry(5.0, "b0"), entry(5.0, "b1")];
    Case::new("a has duplicates", a2, b2.clone(), 0.1).assert_identical_to_reference();
    let a3 = vec![entry(5.0, "a0"), entry(5.0, "a1")];
    let b3 = vec![entry(5.0, "b0"), entry(5.0, "b1"), entry(1.0, "b2")];
    Case::new("b has duplicates", a3, b3, 0.1).assert_identical_to_reference();
}

/// Entries that fall outside `max_dt`, on both sides of the window.
#[test]
fn entries_outside_max_dt() {
    let a = [entry(0.0, "a0"), entry(10.0, "a1"), entry(20.0, "a2")];
    let b = [entry(0.01, "b0"), entry(10.5, "b1"), entry(19.99, "b2")];
    let case = Case::new("one in tolerance, one out", a.to_vec(), b.to_vec(), 0.1);
    case.assert_identical_to_reference();
    assert_eq!(associate(&a, &b, 0.1), vec![(0, 0), (2, 2)]);

    // Entries out of tolerance on the *early* side. Neither sub-case pairs
    // anything at 0.05 s — the minimum difference in each is 0.1 s — so both opt
    // out of the non-empty control rather than asserting a match that does not
    // exist. The comparison still has to agree, and agreement on an empty answer
    // is a real (if weak) result here: the tolerance must not be over-applied.
    let a2 = [entry(0.0, "a0"), entry(1.0, "a1")];
    let b2 = [entry(0.5, "b0"), entry(0.9, "b1")];
    Case::new("everything to the right", a2.to_vec(), b2.to_vec(), 0.05)
        .allowing_empty()
        .assert_identical_to_reference();
    let a3 = [entry(0.5, "a0"), entry(0.9, "a1")];
    let b3 = [entry(0.0, "b0"), entry(1.0, "b1")];
    Case::new("everything to the left", a3.to_vec(), b3.to_vec(), 0.05)
        .allowing_empty()
        .assert_identical_to_reference();

    // Control on each: widening the tolerance to exactly the smallest difference
    // present turns the same lists into a match, so the emptiness above is the
    // 0.05 tolerance's doing and not a blanket early return.
    assert_eq!(
        associate(
            &[entry(0.0, "a0"), entry(1.0, "a1")],
            &[entry(0.5, "b0"), entry(0.9, "b1")],
            0.5
        )
        .len(),
        2
    );
    assert_eq!(
        associate(
            &[entry(0.5, "a0"), entry(0.9, "a1")],
            &[entry(0.0, "b0"), entry(1.0, "b1")],
            0.5
        )
        .len(),
        2
    );
}

/// `max_dt` of exactly zero: only identical timestamps may pair.
#[test]
fn max_dt_zero() {
    let a = [entry(1.0, "a0"), entry(1.0, "a1"), entry(2.0, "a2")];
    let b = [entry(1.0, "b0"), entry(2.0, "b1"), entry(2.0, "b2")];
    let case = Case::new("max_dt = 0", a.to_vec(), b.to_vec(), 0.0);
    case.assert_identical_to_reference();
    assert_eq!(associate(&a, &b, 0.0), vec![(0, 0), (2, 1)]);
}

/// `max_dt` of infinity: every pair is a candidate, so the result is a full
/// one-to-one pairing decided entirely by the difference sort.
#[test]
fn max_dt_infinite() {
    let a = [entry(0.0, "a0"), entry(2.0, "a1"), entry(1.0, "a2")];
    let b = [entry(1.0, "b0"), entry(3.0, "b1"), entry(0.5, "b2")];
    let case = Case::new("max_dt = inf", a.to_vec(), b.to_vec(), f64::INFINITY);
    case.assert_identical_to_reference();
    assert_eq!(associate(&a, &b, f64::INFINITY).len(), 3);

    // Control: the same lists with a finite tolerance drop a pair, so the
    // length above is the infinity, not an accident of the data.
    assert_eq!(associate(&a, &b, 0.75).len(), 2);
}

/// The guard at the top of `associate`: a negative or `NaN` `max_dt` is rejected
/// before any work happens.
#[test]
fn invalid_tolerance_yields_nothing() {
    let a = [entry(0.0, "a0"), entry(1.0, "a1")];
    let b = [entry(0.0, "b0"), entry(1.0, "b1")];
    for bad in [-1.0, -0.5, f64::NAN, f64::NEG_INFINITY] {
        assert!(
            associate(&a, &b, bad).is_empty(),
            "max_dt = {bad:?} must yield no association"
        );
    }
    // -0.0 compares equal to 0.0 and is *not* rejected: the guard is
    // `max_dt < 0.0`, and -0.0 < 0.0 is false. Pinned so a future "tidy up the
    // sign" change cannot silently alter the contract.
    assert_eq!(associate(&a, &b, -0.0).len(), 2);
    // Control: the identical lists at a small positive tolerance do associate.
    assert_eq!(associate(&a, &b, 0.5).len(), 2);
}

/// A non-finite `a` timestamp makes every difference infinite or `NaN`, so it can
/// never satisfy the contract's "the difference is finite" test — at any
/// tolerance, including `+inf`.
#[test]
fn non_finite_a_timestamps_match_nothing() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let a = vec![entry(bad, "a0"), entry(1.0, "a1"), entry(2.0, "a2")];
        let b = vec![entry(1.0, "b0"), entry(2.0, "b1"), entry(3.0, "b2")];
        for tol in [f64::INFINITY, 100.0, 0.5] {
            let case = Case::new("a has a non-finite stamp", a.to_vec(), b.to_vec(), tol);
            case.assert_identical_to_reference();
        }
        // Spelled out: the two finite entries still pair, and the bad one is
        // simply absent from the result.
        let got = associate(&a, &b, f64::INFINITY);
        assert_eq!(
            got,
            vec![(1, 0), (2, 1)],
            "a timestamp {bad:?} must match nothing"
        );
    }
    // A non-finite `b` timestamp likewise. The oracle drops those pairs because
    // their difference is not finite, and the implementation must agree — this is
    // the case a window half-open at `ta + max_dt` cannot get right on its own.
    let a = vec![entry(0.0, "a0"), entry(1.0, "a1")];
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let b = vec![entry(bad, "b0"), entry(0.5, "b1")];
        // A non-finite `b` stamp makes every difference involving it non-finite, so
        // `b[0]` can never be a candidate whatever the tolerance, and the whole
        // result rests on `b[1]` alone. At `max_dt = inf` that leaves exactly one
        // pair; the two finite tolerances opt out of the non-empty control,
        // because there the reference's own answer may legitimately be empty and
        // demanding a match would reject the correct output.
        Case::new(
            "b has a non-finite stamp",
            a.clone(),
            b.clone(),
            f64::INFINITY,
        )
        .assert_identical_to_reference();
        Case::new("b has a non-finite stamp", a.clone(), b.clone(), 0.0)
            .allowing_empty()
            .assert_identical_to_reference();
        Case::new("b has a non-finite stamp", a.clone(), b.clone(), 10.0)
            .allowing_empty()
            .assert_identical_to_reference();
        // The non-finite entry really is excluded rather than merely unmatched by
        // chance: with the finite entry removed, `a` cannot be paired at any
        // tolerance at all, and that is exactly what the reference says.
        let only_bad = vec![entry(bad, "b0")];
        Case::new(
            "b is entirely non-finite",
            a.clone(),
            only_bad.clone(),
            f64::INFINITY,
        )
        .allowing_empty()
        .assert_identical_to_reference();
        assert!(
            associate(&a, &only_bad, f64::INFINITY).is_empty(),
            "a lone non-finite b entry must match nothing"
        );
    }
}

/// The greedy is **global**, not per-`a`: the smallest difference wins even when
/// it belongs to the later `a` entry. A per-`a` greedy — each `a` taking its own
/// closest partner, in `a` order — takes `(0, 0)` here, whose difference is
/// `0.05`, before `(1, 1)`, whose difference is `0.0`.
#[test]
fn greedy_is_global_not_per_a() {
    let a = [entry(0.0, "a0"), entry(2.0, "a1")];
    let b = [entry(0.05, "b0"), entry(1.0, "b1")];
    let case = Case::new("global greedy", a.to_vec(), b.to_vec(), 2.5);
    case.assert_identical_to_reference();
    // Sorted by a's timestamp, so a1 (at 2.0) is reported before a0 (at 0.0).
    assert_eq!(associate(&a, &b, 2.5), vec![(0, 0), (1, 1)]);
}

/// Within one `a` entry's window the differences form a **V**, not a sorted
/// sequence: they fall towards `a`'s own stamp and rise again. An implementation
/// that treated the window as a sorted run and merged the runs would emit the far
/// side before the near side, or miss a pair entirely.
#[test]
fn v_shaped_windows() {
    let a = [entry(1.0, "a0")];
    let b = [
        entry(0.0, "b0"),
        entry(0.5, "b1"),
        entry(0.5, "b2"),
        entry(1.0, "b3"),
        entry(1.0, "b4"),
        entry(1.5, "b5"),
        entry(1.5, "b6"),
    ];
    // The differences are [1.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5]: a V, not a run.
    let case = Case::new("V-shaped window", a.to_vec(), b.to_vec(), 2.0);
    case.assert_identical_to_reference();
    assert_eq!(
        associate(&a, &b, 2.0),
        vec![(0, 3)],
        "the nearest b wins, at diff 0"
    );

    // A second `a` on the far side, so the V's rising arm has to be consulted
    // after the other `a` has taken the nearest entry.
    let a2 = vec![entry(1.0, "a0"), entry(1.6, "a1")];
    let b2 = vec![entry(0.0, "b0"), entry(1.0, "b1"), entry(1.5, "b2")];
    Case::new("V-shaped windows, two a entries", a2, b2, 2.0).assert_identical_to_reference();

    // And one where a consumed `b` forces a *second* `a` to walk past its own
    // nearest, which is exactly the refresh the lazy versions get wrong.
    let a3 = vec![entry(0.0, "a0"), entry(0.0, "a1")];
    let b3 = vec![entry(-0.01, "b0"), entry(0.02, "b1")];
    Case::new("two a entries share a b", a3, b3, 0.05).assert_identical_to_reference();
}

/// Unsorted inputs with a large number of exact ties — the shape that would
/// expose an implementation relying on stability rather than on the documented
/// `(diff, i, j)` order.
#[test]
fn unsorted_inputs_with_ties() {
    let mut rng = Rng::new(0xA55E_C0DE);
    for trial in 0..64 {
        let n = 6 + rng.below(6) as usize;
        let m = 6 + rng.below(6) as usize;
        // A grid of only 3 distinct stamps over the whole range, so ties are
        // rampant.
        let a: Vec<IndexEntry> = (0..n)
            .map(|i| entry((rng.below(3) * 10) as f64, &format!("a{i}")))
            .collect();
        let b: Vec<IndexEntry> = (0..m)
            .map(|i| entry((rng.below(3) * 10) as f64, &format!("b{i}")))
            .collect();
        let max_dt = 20.0;
        let want = associate_reference(&a, &b, max_dt);
        let got = associate(&a, &b, max_dt);
        assert_eq!(
            got,
            want,
            "trial {trial}: a={:?} b={:?}",
            Case::ts(&a),
            Case::ts(&b)
        );
        assert!(!want.is_empty(), "trial {trial} is vacuous");
    }
}

/// A whole random corpus, spanning both tie-heavy quantised data and
/// unstructured real-valued data, over a spread of tolerances.
///
/// The generators are chosen so the corpus contains the shapes that broke the
/// earlier attempts: many `a` entries contending for few `b` (wide contention),
/// long runs sharing one partner, duplicate stamps in both lists, and tolerances
/// wide enough that every cross pair is a candidate.
#[test]
fn randomised_corpus_matches_the_reference() {
    let mut rng = Rng::new(0x1234_5678_9ABC_DEF0);
    let mut total_matches = 0usize;
    let mut cases = 0usize;

    for trial in 0..600 {
        let shape = trial % 6;
        let n = rng.below(14) as usize;
        let m = rng.below(14) as usize;

        let make = |rng: &mut Rng, len: usize| -> Vec<IndexEntry> {
            (0..len)
                .map(|i| {
                    let t = match shape {
                        // Coarse grid: every difference is a small multiple of
                        // the same step, so exact ties abound.
                        0 => (rng.below(4) as f64) * 0.5,
                        // Finer grid.
                        1 => (rng.below(8) as f64) * 0.25,
                        // Real-valued, the way a real `rgb.txt` looks.
                        2 => rng.below(1_000_000) as f64 / 1e6 * 20.0,
                        // Mixed: one value in a small set, one in the continuum.
                        3 => {
                            if rng.below(2) == 0 {
                                (rng.below(3) as f64) * 10.0
                            } else {
                                rng.below(1_000_000) as f64 / 1e6 * 30.0
                            }
                        }
                        // A single repeated stamp per list, with a few outliers:
                        // the extreme contention shape, where nearly every `a`
                        // entry starts out naming the same `b`.
                        4 => {
                            if rng.below(8) == 0 {
                                rng.below(5) as f64
                            } else {
                                1.0
                            }
                        }
                        // Two clusters far apart, so both a's window and b's
                        // insertion-point table straddle a gap.
                        _ => {
                            if rng.below(2) == 0 {
                                (rng.below(6) as f64) * 0.25
                            } else {
                                100.0 + (rng.below(6) as f64) * 0.25
                            }
                        }
                    };
                    entry(t, &format!("e{i}"))
                })
                .collect()
        };

        let a = make(&mut rng, n);
        let b = make(&mut rng, m);

        let max_dt = match trial % 7 {
            0 => 0.0,
            1 => 0.25,
            2 => 1.0,
            3 => 20.0,
            4 => f64::INFINITY,
            5 => 1e9,
            _ => 0.499_999_999_999_999_9,
        };

        let want = associate_reference(&a, &b, max_dt);
        let got = associate(&a, &b, max_dt);
        assert_eq!(
            got,
            want,
            "trial {trial} (shape {shape}, n {n}, m {m}, max_dt {max_dt})\n  a: [{}]\n  b: [{}]",
            Case::ts(&a),
            Case::ts(&b),
        );
        total_matches += want.len();
        cases += 1;
    }

    // Control: the corpus actually exercised both algorithms. A corpus that
    // produced nothing anywhere would compare two empty vectors 600 times.
    assert_eq!(cases, 600);
    assert!(
        total_matches > 1_000,
        "corpus produced only {total_matches} matches across {cases} cases — too \
         weak to distinguish the algorithms"
    );
}

/// Exhaustive over a small alphabet, so no generator can miss a case: every
/// pairing of the two lists below `max_dt` is enumerated for a handful of
/// tolerance values. This is the backstop for the randomised corpus above.
#[test]
fn exhaustive_small_alphabet() {
    // Four stamps with plenty of symmetry, so ties, duplicates and V-shaped
    // windows all occur.
    const STAMPS: [f64; 4] = [0.0, 1.0, 1.0, 3.0];
    let mut rng = Rng::new(0xFEED_FACE);
    let mut pairs_checked = 0usize;

    for trial in 0..500 {
        let n = 1 + rng.below(4) as usize;
        let m = 1 + rng.below(4) as usize;
        let a: Vec<IndexEntry> = (0..n)
            .map(|i| entry(STAMPS[rng.below(4) as usize], &format!("a{i}")))
            .collect();
        let b: Vec<IndexEntry> = (0..m)
            .map(|j| entry(STAMPS[rng.below(4) as usize], &format!("b{j}")))
            .collect();
        for max_dt in [0.0, 1.0, 2.0, 3.0, f64::INFINITY] {
            let want = associate_reference(&a, &b, max_dt);
            let got = associate(&a, &b, max_dt);
            assert_eq!(
                got,
                want,
                "trial {trial} (n {n}, m {m}, max_dt {max_dt})\n  a: [{}]\n  b: [{}]",
                Case::ts(&a),
                Case::ts(&b),
            );
            pairs_checked += 1;
        }
    }
    assert_eq!(
        pairs_checked, 2_500,
        "the sweep above must have run every case"
    );
}

/// `max_dt` sits exactly on a candidate distance, and just off it. `<=` must
/// include the boundary pair; a strict comparison would drop it.
///
/// This is the ulp trap: `ta + max_dt` rounds, so a window half-open at that
/// bound can over-include by an ulp and admit a pair on the far side of
/// `max_dt`. Widening the bound with `next_after` is *worse* — it admits pairs
/// that were never candidates. The boundary has to be decided on the exact
/// difference.
#[test]
fn max_dt_boundary_is_inclusive() {
    let a = [entry(1.0, "a0")];
    let b = [entry(1.5, "b0")];
    assert_eq!(
        associate(&a, &b, 0.5),
        vec![(0, 0)],
        "0.5 == max_dt must match"
    );
    assert!(associate(&a, &b, 0.499_999_999_999_999_9).is_empty());
    Case::new("boundary included", a.to_vec(), b.to_vec(), 0.5).assert_identical_to_reference();
}

/// The same boundary, but with the whole lists around it, so the answer depends
/// on the tolerance test rather than on there being only one candidate.
#[test]
fn max_dt_boundary_with_context() {
    for (ta, tb, tol) in [
        (1.0f64, 1.5f64, 0.5f64),
        (0.1, 0.3, 0.2),
        (1e17, 1e17 + 2.0, 2.0),
        (1.0, 1.0 + f64::EPSILON, f64::EPSILON),
        (1.0, 1.0 - f64::EPSILON, f64::EPSILON),
        // 2^53 and 2^53 + 1 are the same double, so this difference is exactly 0.
        (9_007_199_254_740_992.0, 9_007_199_254_740_992.0, 0.0),
    ] {
        // The "one ulp under" variant is allowed to be empty, because a tolerance of
        // exactly `0.0` steps down to `-0.0`, which the guard admits — and which
        // still pairs every coincident pair, since `0.0 <= -0.0`. Demanding an
        // empty answer there would reject the correct output. The variant
        // therefore only has to agree with the reference.
        //
        // `wrapping_sub` rather than `- 1`: `0.0f64.to_bits()` is 0, and a plain
        // subtraction panics on overflow in a debug build.
        let a = vec![entry(ta, "a0"), entry(ta + 100.0, "a1")];
        let b = vec![entry(tb, "b0"), entry(tb + 100.0, "b1")];
        Case::new("boundary with context", a.clone(), b.clone(), tol)
            .assert_identical_to_reference();
        Case::new(
            "boundary with context, one ulp under",
            a,
            b,
            f64::from_bits(tol.to_bits().wrapping_sub(1)),
        )
        .allowing_empty()
        .assert_identical_to_reference();
    }
}

// ---------------------------------------------------------------------------
// The scaling assertion
// ---------------------------------------------------------------------------

fn peak_rss_bytes() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").expect("/proc/self/status");
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kib: u64 = rest
                .trim()
                .trim_end_matches(" kB")
                .trim()
                .parse()
                .expect("VmHWM is a number");
            return kib * 1024;
        }
    }
    panic!("no VmHWM in /proc/self/status");
}

/// Build `n` entries per list whose stamps are quantised onto a grid finer than
/// `max_dt`, so that every one of the `n * n` cross pairs is a candidate — the
/// input shape that made the previous implementation build its whole table.
///
/// (`step` is derived from `n` in the caller, not fixed here, so `b`'s stamps
/// stay distinct and the pairing is a bijection rather than one degenerate
/// clump.)
fn dense_pair(n: usize, max_dt: f64) -> (Vec<IndexEntry>, Vec<IndexEntry>) {
    let step = max_dt / (2.0 * n as f64);
    let a = (0..n)
        .map(|k| {
            // A's stamps collapse onto a coarser 10x grid: many exact ties.
            let t = (k as f64 * step * 10.0).trunc() / 10.0;
            entry(t, &format!("rgb/{k}.png"))
        })
        .collect();
    let b = (0..n)
        .map(|k| entry(k as f64 * step, &format!("depth/{k}.png")))
        .collect();
    (a, b)
}

/// The regression test: at a size where the old implementation was already
/// minutes and gigabytes of work, the new one must stay near linear.
///
/// The assertion is on **peak RSS**, not on wall clock — a timing threshold is
/// flaky under load and would be measuring the machine, not the algorithm.
/// What is being ruled out is the `O(n * m)` table itself, and peak RSS is a
/// direct witness of whether one was built: 20 000 x 20 000 pairs would be
/// 400 million `24`-byte tuples, ~9.6 GB, against a few MiB for the two index
/// vectors, the run buffers and the result.
#[test]
#[cfg_attr(not(target_os = "linux"), ignore)]
fn dense_matches_peak_rss_stays_linear() {
    const N: usize = 20_000;
    const MAX_DT: f64 = 0.02;

    let (a, b) = dense_pair(N, MAX_DT);

    // Confirm the input really is the adversarial shape: every cross pair is a
    // candidate. This is the property that made the old code's table `O(n*m)`,
    // and it is asserted rather than assumed, so the test cannot quietly stop
    // covering the case it exists for.
    let candidates: usize = a
        .iter()
        .map(|ea| {
            b.iter()
                .filter(|eb| (ea.timestamp - eb.timestamp).abs() <= MAX_DT)
                .count()
        })
        .sum();
    assert_eq!(
        candidates,
        N * N,
        "the input is supposed to have every cross pair in tolerance"
    );

    let before = peak_rss_bytes();
    let started = std::time::Instant::now();
    let matches = associate(&a, &b, MAX_DT);
    let elapsed = started.elapsed();
    let after = peak_rss_bytes();
    let growth = after.saturating_sub(before);

    // The result itself: one pair per entry, and every distance in tolerance.
    // These are the correctness controls — the RSS bound below would also be
    // satisfied by a function that returned nothing.
    assert_eq!(
        matches.len(),
        N,
        "every entry should be paired exactly once"
    );
    let mut a_seen = std::collections::BTreeSet::new();
    let mut b_seen = std::collections::BTreeSet::new();
    for &(i, j) in &matches {
        assert!(a_seen.insert(i), "a index {i} used twice");
        assert!(b_seen.insert(j), "b index {j} used twice");
        let d = (a[i].timestamp - b[j].timestamp).abs();
        assert!(d <= MAX_DT, "pair ({i}, {j}) is {d} apart, beyond {MAX_DT}");
    }

    // The quadratic table for this input is 400 million x 24 bytes = 9.6 GB.
    // Allow 128 MiB, which is >100x the few MiB the implementation actually needs
    // and still an order of magnitude below the smallest quadratic footprint that
    // could exist here. The 24-byte figure is hard-coded from the old
    // `Vec<(f64, usize, usize)>` layout; on any target where a tuple is
    // smaller, the bound only gets looser, which is the safe direction.
    const BUDGET: u64 = 128 * 1024 * 1024;
    assert!(
        growth < BUDGET,
        "associate() grew peak RSS by {:.1} MiB at n = {N} with all {candidates} \
         cross pairs in tolerance; the budget is {} MiB. A non-linear growth here \
         means the cross product is being materialised again.",
        growth as f64 / (1024.0 * 1024.0),
        BUDGET / (1024 * 1024),
    );

    // Note: the oracle is deliberately not run at this size. It is O(n*m) by
    // construction and would be the very thing being tested.
    eprintln!(
        "dense n={N}: {candidates} candidate pairs, {} matches, peak RSS growth {:.1} MiB (budget {} MiB), associate() took {:.3?}",
        matches.len(),
        growth as f64 / (1024.0 * 1024.0),
        BUDGET / (1024 * 1024),
        elapsed,
    );
}

// ---------------------------------------------------------------------------
// End-to-end: through the real file reader
// ---------------------------------------------------------------------------

/// `associate` fed by `read_index`, the way a dataset sequence is loaded —
/// unsorted, comment-bearing files on disk.
#[test]
fn association_over_files_read_from_disk() {
    let dir = std::env::temp_dir().join("tum_associate_scaling");
    std::fs::create_dir_all(&dir).expect("temp dir");

    // Deliberately out of timestamp order, with comments and blank lines, and
    // with several exact ties.
    let rgb = dir.join(format!("rgb_{}.txt", unique_id()));
    let depth = dir.join(format!("depth_{}.txt", unique_id()));
    std::fs::write(
        &rgb,
        "# color images\n\
         # file: rgb.txt\n\
         \n\
         1305031102.375304 rgb/c.png\n\
         1305031102.175304 rgb/a.png\n\
         1305031102.275326 rgb/b.png\n\
         1305031102.375304 rgb/d.png\n",
    )
    .expect("write rgb");
    std::fs::write(
        &depth,
        "# depth images\n\
         1305031102.175304 depth/a.png\n\
         1305031102.280000 depth/x.png\n\
         1305031102.375304 depth/b.png\n",
    )
    .expect("write depth");

    let a = cv_io::datasets::tum::read_index(&rgb).expect("read rgb.txt");
    let b = cv_io::datasets::tum::read_index(&depth).expect("read depth.txt");
    assert_eq!(a.len(), 4);
    assert_eq!(b.len(), 3);

    let got = associate(&a, &b, 0.05);
    let want = associate_reference(&a, &b, 0.05);
    assert_eq!(got, want);
    assert!(
        !want.is_empty(),
        "vacuous: the files produced no association to compare"
    );

    // Control: the result is ordered by a's timestamp, so rgb/a.png (the earliest
    // a-entry) comes first, and every one of the three depth frames is matched —
    // their nearest rgb frame is 0.000, 0.0047 and 0.000 s away respectively, all
    // inside 0.05 s.
    //
    // The earlier revision of this file asserted a length of 2 here, on the
    // theory that only two depth frames were in tolerance. That was wrong:
    // 1305031102.375304 is 0.0047 s from depth/x.png as well as being an exact
    // match for depth/b.png, and it is the greedy that decides between the two,
    // not the window. The assertion now states what the timestamps actually say.
    assert_eq!(
        got.first().map(|p| a[p.0].filename.clone()),
        Some("rgb/a.png".to_owned())
    );
    assert_eq!(
        got,
        vec![(1, 0), (2, 1), (0, 2)],
        "all three depth frames are within 0.05 s of some rgb frame"
    );

    // The a-entry furthest from everything (rgb/a.png at 1305031102.175304 is
    // 0.1047 s from depth/x.png) is still matched, to its exact partner - which is
    // the point: `b`'s own nearest, not the nearest that happens to be listed
    // first.
    assert!(got.contains(&(1, 0)));

    // Control for the reader path itself: a tolerance too tight for the offset
    // frames still leaves the *exact* matches, which are rgb/a.png with
    // depth/a.png and rgb/c.png (or rgb/d.png) with depth/b.png. So 1e-6 keeps
    // 2 pairs rather than the 3 that 0.05 s allows: the offset frames drop out and
    // the coincidences do not.
    let tight = associate(&a, &b, 1e-6);
    assert_eq!(
        tight,
        vec![(1, 0), (0, 2)],
        "only the exact matches survive"
    );
    // And a tolerance of literally zero keeps only the coincident pairs too.
    assert_eq!(associate(&a, &b, 0.0), vec![(1, 0), (0, 2)]);
}
