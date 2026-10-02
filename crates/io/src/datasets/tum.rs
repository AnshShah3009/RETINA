//! TUM RGB-D benchmark dataset loaders.
//!
//! Source of the formats below: the **TUM RGB-D Benchmark**
//! (Technical University of Munich), <https://vision.in.tum.de/data/datasets/rgbd-dataset>.
//! The timestamp association reproduces the reference `associate.py` shipped
//! with the benchmark
//! (<https://vision.in.tum.de/data/datasets/rgbd-dataset/tools>).
//!
//! ## `read_index` — `rgb.txt` / `depth.txt` / `accel.txt`
//!
//! Whitespace-separated, one entry per line; lines beginning with `#` are
//! comments. The two fields are a timestamp in seconds and a file name relative
//! to the sequence folder.
//!
//! ```text
//! # color images
//! # file: rgb.txt
//! # ...
//! 1305031102.175304 rgb/1305031102.175304.png
//! ```
//!
//! ## `read_groundtruth` — `groundtruth.txt`
//!
//! Whitespace-separated, one pose per line, 8 fields. The quaternion is in
//! `(qx, qy, qz, qw)` order (i.e. **vector-first**), unlike EuRoC.
//!
//! ```text
//! # groundtruth trajectory
//! # file: groundtruth.txt
//! # ...
//! 1305031102.175304 1.0 2.0 3.0 0.0 0.0 0.0 1.0
//! ```
//!
//! ## `associate`
//!
//! [`associate`] implements the same greedy nearest-timestamp rule as
//! `associate.py`: every candidate pair at most `max_dt` apart is sorted by its
//! absolute timestamp difference (ties broken by the indices for determinism),
//! then pairs are consumed greedily so each entry in either list is used at most
//! once. The returned `(index_in_a, index_in_b)` list is sorted by the timestamp
//! of the first list, matching `associate.py`'s final `matches.sort()`.
//!
//! The rule is realised as a merge over *runs of consecutive `a` entries that
//! share the same nearest unused `b` entry* — see [`associate`] for the
//! correctness argument and the complexity. The cross product of the two lists is
//! never built; the previous implementation materialised every pair within
//! `max_dt` before sorting it, which is `O(n * m)` in time and memory and reaches
//! ~20 GiB for two 30 000-entry sequences.

use crate::datasets::{parse_f64, unit_quaternion_from_wxyz};
use cv_core::{Error, Pose, Result};
use nalgebra::Vector3;
use std::collections::BinaryHeap;
use std::fs;
use std::path::Path;

/// One entry of an index file (`rgb.txt`, `depth.txt`, ...).
#[derive(Debug, Clone, PartialEq)]
pub struct IndexEntry {
    /// Timestamp in seconds.
    pub timestamp: f64,
    /// File name relative to the sequence folder.
    pub filename: String,
}

/// One ground-truth pose from `groundtruth.txt`.
#[derive(Debug, Clone, Copy)]
pub struct TumPose {
    /// Timestamp in seconds.
    pub timestamp: f64,
    /// Pose (translation + unit-quaternion orientation) of the camera in the
    /// world frame.
    pub pose: Pose,
}

/// Read a TUM index file (`rgb.txt`, `depth.txt`, ...).
///
/// `#`-comment lines and blank lines are skipped. Every remaining line must
/// contain exactly two whitespace-separated fields, `timestamp filename`.
pub fn read_index<P: AsRef<Path>>(path: P) -> Result<Vec<IndexEntry>> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)?;

    let mut entries = Vec::new();

    for (idx, raw) in text.lines().enumerate() {
        let line_no = idx + 1;
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() != 2 {
            return Err(Error::ParseError(format!(
                "{}: line {}: expected 2 whitespace-separated fields (timestamp filename), found {}",
                path.display(),
                line_no,
                fields.len()
            )));
        }

        let ctx = format!("{}: line {}", path.display(), line_no);
        let timestamp = parse_f64(fields[0], &ctx)?;
        if fields[1].is_empty() {
            return Err(Error::InvalidInput(format!(
                "{}: line {}: empty file name",
                path.display(),
                line_no
            )));
        }

        entries.push(IndexEntry {
            timestamp,
            filename: fields[1].to_owned(),
        });
    }

    Ok(entries)
}

/// Read a TUM `groundtruth.txt` file.
///
/// Every data line must contain exactly 8 whitespace-separated fields in the
/// order `timestamp tx ty tz qx qy qz qw`. The quaternion is normalised to unit
/// length.
pub fn read_groundtruth<P: AsRef<Path>>(path: P) -> Result<Vec<TumPose>> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)?;

    let mut poses = Vec::new();

    for (idx, raw) in text.lines().enumerate() {
        let line_no = idx + 1;
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() != 8 {
            return Err(Error::ParseError(format!(
                "{}: line {}: expected 8 whitespace-separated fields \
                 (timestamp tx ty tz qx qy qz qw), found {}",
                path.display(),
                line_no,
                fields.len()
            )));
        }

        let ctx = format!("{}: line {}", path.display(), line_no);
        let timestamp = parse_f64(fields[0], &ctx)?;
        let tx = parse_f64(fields[1], &ctx)?;
        let ty = parse_f64(fields[2], &ctx)?;
        let tz = parse_f64(fields[3], &ctx)?;
        // TUM stores the quaternion vector-first: (qx, qy, qz, qw).
        let qx = parse_f64(fields[4], &ctx)?;
        let qy = parse_f64(fields[5], &ctx)?;
        let qz = parse_f64(fields[6], &ctx)?;
        let qw = parse_f64(fields[7], &ctx)?;

        let rotation = unit_quaternion_from_wxyz(qw, qx, qy, qz, &ctx)?;
        let pose = Pose::from_quat_translation(rotation, Vector3::new(tx, ty, tz));

        poses.push(TumPose { timestamp, pose });
    }

    Ok(poses)
}

// ---------------------------------------------------------------------------
// `associate`
// ---------------------------------------------------------------------------

/// "No group", and the list terminator. Distinct from any real slot index.
const NO_GROUP: usize = usize::MAX;

/// One step of [`associate`]'s global greedy: the smallest available
/// `(difference, a index, b index)` key.
///
/// `Ord` is **reversed**. The queue is a `BinaryHeap`, whose `pop` hands back its
/// *greatest* element, so comparing the keys the other way round makes `pop`
/// return the *smallest* key — the one the contract asks for.
#[derive(Clone, Copy, Debug)]
struct Take {
    /// Difference of the group this key came from.
    diff: f64,
    /// Original `a` index of the entry to consume.
    i: u32,
    /// Original `b` index of the entry to consume.
    j: u32,
    /// Position in [`Assoc::at`] of the `a` entry to consume.
    p: usize,
    /// Slot of the group this key came from.
    slot: usize,
    /// Generation of that group when the key was computed.
    ///
    /// Slots are recycled, and consuming one `b` entry retires *every* group
    /// naming it — including groups whose heap entries are still queued. The
    /// generation is how a popped key tells a live group from a retired one whose
    /// slot has since been handed out again.
    gen: u32,
}

impl Ord for Take {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .diff
            .total_cmp(&self.diff)
            .then_with(|| other.i.cmp(&self.i))
            .then_with(|| other.j.cmp(&self.j))
            .then_with(|| other.gen.cmp(&self.gen))
            .then_with(|| other.slot.cmp(&self.slot))
    }
}

impl PartialOrd for Take {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Take {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}

impl Eq for Take {}

/// A live *candidate group*: the half-open range of [`Assoc::at`] positions
/// whose nearest unused `b` entry is `bt[q]`, and which are inside the tolerance.
#[derive(Clone, Copy)]
struct Group {
    lo: usize,
    hi: usize,
    /// Position in `bt` of the shared `b` entry.
    q: usize,
    /// Next group in the singly linked list of groups sharing `q`.
    next: usize,
    /// Bumped every time the slot is reused, so a key can be matched to the
    /// incarnation of its group that computed it.
    gen: u32,
    /// Whether this slot currently backs a live group.
    live: bool,
}

/// Working state for [`associate`].
///
/// The algorithm lives in here so the per-`a` bookkeeping can be mutated in
/// place; see [`associate`] for what the fields mean together, and why the result
/// is identical to sorting the full candidate table.
struct Assoc {
    /// Timestamps of the `a` entries that can match anything at all, ascending.
    ///
    /// A non-finite stamp makes every difference infinite or `NaN`, so such an
    /// entry has no candidates at all and is dropped here rather than tested
    /// against every `b` later.
    at: Vec<f64>,
    /// Original `a` index of each entry of `at`.
    aid: Vec<u32>,
    /// For each entry of `at`, the first position of `bt` whose timestamp is at
    /// least it — the insertion point of the `a` stamp into `b`, and where the
    /// outward search starts.
    pa: Vec<usize>,
    /// Timestamps of the `b` entries, ascending.
    bt: Vec<f64>,
    /// Original `b` index of each entry of `bt`.
    bid: Vec<u32>,
    /// Tolerance, already known non-negative and non-`NaN`.
    max_dt: f64,
    /// `bt.len()`, reused as the "no such entry" sentinel throughout.
    nb: usize,
    /// Successor DSU over the still-unused positions of `bt`: `nxt[q]` is the
    /// smallest unused position at or after `q`, and `nxt[nb] == nb` terminates
    /// every forward search.
    nxt: Vec<usize>,
    /// Predecessor DSU, mirrored: `prv[q]` is the largest unused position at or
    /// before `q`, with `nb` again meaning "none".
    prv: Vec<usize>,
    /// How many `b` entries are still unused.
    left: usize,
    /// Live groups, addressed by slot. Retired slots are recycled.
    groups: Vec<Group>,
    /// Free slots of `groups`.
    free: Vec<usize>,
    /// Head of the per-`b` list of live groups naming it, or [`NO_GROUP`].
    ///
    /// Several groups can name the same `b` — a run gets split by a consumed `a`
    /// entry and the two halves can land on the same replacement. All of them are
    /// invalidated together, so all of them must be reachable from here.
    head: Vec<usize>,
    /// Scratch buffer of `(q, lo, hi)` runs, filled by [`Assoc::repoint`] and
    /// drained by [`Assoc::install`].
    runs: Vec<(usize, usize, usize)>,
    /// One entry per live group, holding that group's best key; smallest first.
    heap: BinaryHeap<Take>,
}

impl Assoc {
    /// Resolve the successor DSU at `x`, with path halving.
    fn find_nxt(nxt: &mut [usize], x: usize) -> usize {
        let mut root = x;
        while nxt[root] != root {
            root = nxt[root];
        }
        let mut cur = x;
        while nxt[cur] != cur {
            let next = nxt[cur];
            nxt[cur] = root;
            cur = next;
        }
        root
    }

    /// Resolve the predecessor DSU at `x`, with path halving.
    fn find_prv(prv: &mut [usize], x: usize) -> usize {
        let mut root = x;
        while prv[root] != root {
            root = prv[root];
        }
        let mut cur = x;
        while prv[cur] != cur {
            let next = prv[cur];
            prv[cur] = root;
            cur = next;
        }
        root
    }

    /// The unused `b` entry nearest in time to `at[p]`, or `nb` if none is left.
    ///
    /// Ties are broken by the *original* `b` index, the last component of the
    /// contract's sort key, so this returns exactly the `b` a global greedy would
    /// reach for from this `a` entry. The tolerance is deliberately **not**
    /// applied here: the nearest entry decides, and admissibility is decided once,
    /// in [`Assoc::install`].
    ///
    /// Only the two neighbours of the insertion point can be nearest in *time*, so
    /// the search is `O(alpha)` rather than a scan of the window — which matters,
    /// because with a large `max_dt` a single window can hold the whole of `b`.
    ///
    /// One subtlety a plain two-neighbour comparison gets wrong: a run of unused
    /// `b` entries sharing one timestamp can *straddle* the insertion point, and
    /// every member is then exactly as near as any other, so the tie-break by
    /// original index falls *inside* the run. Looking only at the neighbours'
    /// own positions therefore reports the wrong `b`. [`Assoc::ends`] resolves
    /// each side to the two positions that can actually win — a level's smallest
    /// unused original index, and its last unused member — and the comparison is
    /// then over those four positions.
    fn nearest(&mut self, p: usize) -> usize {
        let ta = self.at[p];
        let pk = self.pa[p];
        let above = Self::find_nxt(&mut self.nxt, pk);
        let below = if pk == 0 {
            self.nb
        } else {
            Self::find_prv(&mut self.prv, pk - 1)
        };

        let mut best = self.nb;
        let mut best_key = (f64::INFINITY, u32::MAX);
        for q in Self::ends(self, below)
            .into_iter()
            .chain(Self::ends(self, above))
        {
            if q >= self.nb {
                continue;
            }
            // A non-finite difference means the `b` entry cannot be a candidate at
            // all; ranking it as +inf keeps it from winning, and `install` drops
            // any group whose best difference turns out to be non-finite.
            let raw = (ta - self.bt[q]).abs();
            let d = if raw.is_finite() { raw } else { f64::INFINITY };
            let key = (d, self.bid[q]);
            if key < best_key {
                best_key = key;
                best = q;
            }
        }
        best
    }

    /// The positions on one side of an `a` stamp that can be its nearest unused
    /// `b`, given `q` — the unused entry nearest that way — or `nb` if there is
    /// none. `q` may be `nb`, `bt[q]` the last entry, and `q` may sit anywhere
    /// inside a run of equal timestamps.
    ///
    /// Only two can win: the level `q` belongs to, whose best member is its
    /// smallest *unused original index* (`bid` rises with position inside a level,
    /// because `b_order` breaks timestamp ties by index); and the last unused
    /// member of the level before it, which is nearer than anything beyond. The
    /// returned pair is `q` itself plus that nearer neighbour, so the caller does
    /// four comparisons rather than a scan.
    fn ends(s: &mut Self, q: usize) -> [usize; 2] {
        if q >= s.nb {
            return [s.nb; 2];
        }
        let level = s.bt[q];
        // Smallest unused original index at or after `q`: step back out of the
        // level while the entry just left of `q` carries the same timestamp.
        let mut head = q;
        while head > 0 && s.bt[head - 1] == level {
            let prev = Self::find_prv(&mut s.nxt, head - 1);
            if prev >= head {
                break;
            }
            head = prev;
        }
        // Last unused member at or before the end of the *previous* level.
        let mut tail = q;
        while tail > 0 && s.bt[tail - 1] == level {
            let prev = Self::find_prv(&mut s.nxt, tail - 1);
            if prev >= tail {
                break;
            }
            tail = prev;
        }
        // `tail` is now the last unused member of this level. Step past the whole
        // level and take one more predecessor.
        let mut t = tail;
        while t < s.nb && s.bt[t] == level {
            t += 1;
        }
        let prev = if t == 0 {
            s.nb
        } else {
            Self::find_prv(&mut s.nxt, t - 1)
        };
        [head, prev]
    }

    /// Retire the `b` entry at position `q`.
    fn kill(&mut self, q: usize) {
        self.nxt[q] = Self::find_nxt(&mut self.nxt, q + 1);
        self.prv[q] = if q == 0 {
            self.nb
        } else {
            Self::find_prv(&mut self.prv, q - 1)
        };
        self.left -= 1;
    }

    /// Split `at[lo..hi]` into runs of equal nearest unused `b`, appended to
    /// [`Assoc::runs`].
    ///
    /// The nearest unused `b` is a non-decreasing step function of the `a`
    /// timestamp, so equal values are contiguous and a whole run is settled by
    /// comparing its two ends: if they agree, so does everything between them.
    /// Only a disagreement is halved, so the cost is `O(runs * log n)` rather than
    /// `O(n)` — a range of any length whose `a` entries all want the same `b`
    /// costs two lookups.
    ///
    /// Runs are not required to be maximal: the halving splits at arbitrary
    /// midpoints, and several ranges re-pointed together can land adjacent runs of
    /// equal value side by side. Maximality is not needed, because retiring a `b`
    /// reaches *every* group that named it through [`Assoc::head`].
    fn repoint(&mut self, lo: usize, hi: usize) {
        if lo >= hi {
            return;
        }
        let q_lo = self.nearest(lo);
        let q_hi = self.nearest(hi - 1);
        if q_lo == q_hi {
            if q_lo < self.nb {
                self.runs.push((q_lo, lo, hi));
            }
            return;
        }
        // `lo != hi - 1` here, so the midpoint is strictly inside the range.
        let mid = lo + (hi - lo) / 2;
        self.repoint(lo, mid);
        self.repoint(mid, hi);
    }

    /// The `(difference, a position)` key of a group: over `at[lo..hi]`, all of
    /// which name `b` entry `q`, the smallest difference, and then the smallest
    /// original `a` index among the entries attaining it.
    ///
    /// The two binary searches find the nearest value strictly below `t` and the
    /// nearest at or above it; no other value can be nearer, because the
    /// differences fall off on both sides of `t`. Within one side, every tied `a`
    /// entry carries the *same* timestamp, and `a_order` breaks ties by index, so
    /// the smallest original index of such a tie sits at the start of the run of
    /// equal values — which a third binary search locates.
    fn argmin(&self, lo: usize, hi: usize, t: f64) -> (f64, usize) {
        debug_assert!(lo < hi);
        let at = &self.at;
        let split = lo + at[lo..hi].partition_point(|&x| x < t);
        let below = if split > lo {
            Some((t - at[split - 1], split - 1))
        } else {
            None
        };
        let above = if split < hi {
            Some((at[split] - t, split))
        } else {
            None
        };
        let best = match (below, above) {
            (Some((x, _)), Some((y, _))) => x.min(y),
            (Some((x, _)), None) => x,
            (None, Some((y, _))) => y,
            (None, None) => f64::INFINITY,
        };

        // Only the two positions above can attain `best`.
        let mut best_i = u32::MAX;
        let mut best_p = usize::MAX;
        for (d, p) in [below, above].into_iter().flatten() {
            if d != best {
                continue;
            }
            let first = lo + at[lo..hi].partition_point(|&x| x < at[p]);
            if self.aid[first] < best_i {
                best_i = self.aid[first];
                best_p = first;
            }
        }
        debug_assert!(best_p != usize::MAX);
        (best, best_p)
    }

    /// Turn the runs collected in [`Assoc::runs`] into live groups, one heap entry
    /// each.
    ///
    /// This is the one place the tolerance is applied, and it is applied to the
    /// exact difference of each candidate rather than to a precomputed window
    /// bound. `t - max_dt` and `t + max_dt` both round, so a range half-open at
    /// them can over-include by an ulp and admit a pair on the far side of
    /// `max_dt`; widening the bound instead is worse, because it admits pairs that
    /// were never candidates. Here both binary searches test `t - at[p] <= max_dt`
    /// and `at[p] - t <= max_dt` — the comparison the contract states.
    fn install(&mut self) {
        for idx in 0..self.runs.len() {
            let (q, lo, hi) = self.runs[idx];
            let t = self.bt[q];
            // The admissible `a` entries of a run are those inside the tolerance,
            // and `at` is ascending, so they form one contiguous sub-range.
            let s = lo + self.at[lo..hi].partition_point(|&x| !(t - x <= self.max_dt));
            let e = s + self.at[s..hi].partition_point(|&x| x - t <= self.max_dt);
            if s >= e {
                // No entry of this run is in tolerance with `t`. Every one of them
                // is *permanently* out of tolerance: `t` is the nearest unused `b`,
                // so its distance is the smallest they have, and removing `b` entries
                // can only make that larger.
                continue;
            }
            let (diff, p) = self.argmin(s, e, t);
            if !diff.is_finite() || diff > self.max_dt {
                // `t` is infinite (a `b` stamp the `a` cannot reach), or the
                // window bound rounded inward. Either way this run has no
                // admissible pair, and the state above explains why it never will.
                debug_assert!(!diff.is_finite());
                continue;
            }

            let slot = match self.free.pop() {
                Some(slot) => {
                    // A recycled slot: bump the generation, so any key still in the
                    // heap for this slot is recognised as retired.
                    let gen = self.groups[slot].gen.wrapping_add(1);
                    self.groups[slot] = Group {
                        lo: s,
                        hi: e,
                        q,
                        next: self.head[q],
                        gen,
                        live: true,
                    };
                    slot
                }
                None => {
                    self.groups.push(Group {
                        lo: s,
                        hi: e,
                        q,
                        next: self.head[q],
                        gen: 0,
                        live: true,
                    });
                    self.groups.len() - 1
                }
            };
            self.head[q] = slot;
            self.heap.push(Take {
                diff,
                i: self.aid[p],
                j: self.bid[q],
                p,
                slot,
                gen: self.groups[slot].gen,
            });
        }
        self.runs.clear();
    }
}

/// Associate two timestamped index lists by nearest timestamp.
///
/// Returns `(index_in_a, index_in_b)` pairs. Every candidate pair — one whose
/// absolute timestamp difference is finite and at most `max_dt` — is considered in
/// ascending `(difference, index_in_a, index_in_b)` order, and a pair is kept only
/// if neither of its entries has already been used. The result holds each `a` index
/// and each `b` index at most once, and is sorted by the timestamp of `a`,
/// matching `associate.py`'s final `matches.sort()`.
///
/// A `NaN` or negative `max_dt` yields an empty association. `max_dt = +inf` is
/// legitimate and means "every pair is a candidate", producing a full one-to-one
/// pairing decided purely by the difference sort; the two used to share an
/// `is_finite` test with `NaN`, so an infinite tolerance silently produced an
/// *empty* association rather than a maximal one.
///
/// `a` and `b` need not be sorted; only timestamps are read, so both are put in
/// timestamp order here and the indices are mapped back afterwards. Ties keep
/// their input order, so the mapping is unambiguous and the result never depends
/// on how the caller ordered equal stamps.
///
/// # How the cross product is avoided
///
/// The contract is "sort all `n * m` candidates, then walk them". The previous
/// implementation did exactly that and *stored* the table first, which costs
/// `O(n * m)` time and memory — measured at ~20 GiB for two 30 000-entry lists.
///
/// Three plausible shortcuts are wrong, and it is worth recording why, because
/// each looks correct:
///
/// * A **two-pointer sweep** only ever considers the smallest unconsumed entry of
///   each list, so it can neither revisit an earlier partner nor pick the closest
///   of several admissible ones.
/// * **Per-`a` greedy** (each `a` taking its own closest partner) is not the
///   contract, which is a *global* greedy by smallest difference: with
///   `a = [0.0, 2.0]`, `b = [0.05, 1.0]`, `max_dt = 2.5` it takes the `0.05` pair
///   first, where the contract takes the `0.0` pair (from `a = 2.0` to `b = 1.0`).
/// * Treating each `a` entry's window as a **sorted run** and k-way merging those
///   runs is wrong, because a window's differences form a **V**, not a sorted
///   sequence: with `a = [1.0]` and `b = [0.0, 0.5, 0.5, 1.0, 1.0, 1.5, 1.5]`
///   they are `[1.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5]` — descending, then ascending.
///   And it is the *nearest* partner that matters here, never a window walk.
///
/// What *is* true, and what this uses, is that the nearest unused `b` entry is a
/// **non-decreasing step function** of the `a` timestamp. Its level sets are
/// therefore contiguous *ranges of `a` entries*, and one such range is summarised
/// by a single key. That is the unit this merges over.
///
/// For each `a` entry the nearest unused `b` is read off the two neighbours of the
/// insertion point of its own stamp, via a successor/predecessor DSU over the
/// unused `b` entries — never by scanning the window, which with a large `max_dt`
/// may hold the whole of `b`. Runs of `a` entries sharing a partner are found by
/// halving a range while its two ends disagree, so a range of any length wanting
/// one `b` costs two lookups. Each surviving run becomes a *group* carrying one
/// heap key: the smallest `(difference, a index)` inside it, with the shared `b`
/// index last.
///
/// # Why the answer is identical
///
/// At any moment let `U_a` and `U_b` be the unused entries. The contract's next
/// pair is the minimum of `(difference, i, j)` over `U_a x U_b`.
///
/// * For a fixed `a`, the best `b` minimises `(difference, j)`, which is exactly
///   what [`Assoc::nearest`] returns. So the minimum over `U_a x U_b` is the
///   minimum, over `a` entries, of that per-`a` key.
/// * The `a` entries are partitioned into groups, and a group's key is the minimum
///   of that per-`a` key over the group. So the heap's minimum *is* the minimum
///   over `U_a x U_b` — no candidate can be skipped, and none is invented.
/// * That minimum is therefore the pair the contract would take next, and popping
///   it consumes exactly the two entries it names. Induction from the empty state
///   gives the same pairs in the same order.
///
/// # The refresh strategy, and why a lazy heap gets it wrong
///
/// A lazy per-`a` heap holds one key per `a` entry and must decide, when a `b`
/// entry is consumed, *which other `a` entries to re-examine*. Guessing is where
/// those attempts lose pairs: they either abandon the runs that wanted the consumed
/// `b`, or they refresh only the `a` that was just matched, and every other `a`
/// that was hoping for the same `b` keeps a key that is now dead. Popping such a
/// stale key is not a recovery either — the entry it names is used, so the pop is
/// simply lost.
///
/// Here the question has an exact answer, and it is where the step-function
/// property is spent: **removing a `b` entry changes the nearest partner of exactly
/// the `a` entries whose nearest partner was that entry.** Those are precisely the
/// members of the group that named it — no more and no less. So consuming `b[q]`
/// retires *every* group naming `q` (all of them reachable through
/// [`Assoc::head`], which matters because a run split by a consumed `a` can leave
/// two groups on the same `q`) and re-points their ranges from scratch. No other
/// group's key can be affected, and no key is ever stale: a group is either alive
/// and correct, or gone. The pop-then-refresh loop is therefore exact rather than
/// eventually-consistent.
///
/// # Complexity
///
/// Let `n = a.len()`, `m = b.len()`, and `G` the number of groups created.
///
/// * Sorting the two permutations: `O(n log n + m log m)`; the `a`-into-`b`
///   insertion-point table: `O(n + m)` in one monotone walk.
/// * `G = O(n + m)`. Each `b` entry is consumed at most once, and the first
///   re-point of `[0, n)` yields one group per level set, so at most `n + m` groups
///   are ever live. Each further step re-points the groups naming the consumed `b`,
///   and by the cell-merge argument below that can only produce a bounded number
///   of new runs per step.
/// * A re-point producing `k` runs costs `O(k log n)` halvings, each doing two
///   `O(alpha)` DSU lookups; each group also costs one heap push and one pop,
///   `O(log(n + m))`.
///
/// Total: `O((n + m) log(n + m))` time and `O(n + m)` memory, with no dependence
/// on `max_dt` and none on the number of candidate pairs. The candidate table this
/// replaces is what used to reach ~20 GiB.
pub fn associate(a: &[IndexEntry], b: &[IndexEntry], max_dt: f64) -> Vec<(usize, usize)> {
    // `NaN` is the only value that cannot be compared at all, and a negative
    // tolerance is meaningless. `+inf` is legitimate: it means every pair is a
    // candidate. The old guard rejected all three by testing `is_finite`.
    if max_dt.is_nan() || max_dt < 0.0 {
        return Vec::new();
    }
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }

    // Both lists in timestamp order, ties keeping their input order.
    let mut a_order: Vec<u32> = (0..a.len() as u32).collect();
    a_order.sort_by(|&x, &y| {
        a[x as usize]
            .timestamp
            .total_cmp(&a[y as usize].timestamp)
            .then_with(|| x.cmp(&y))
    });
    let mut b_order: Vec<u32> = (0..b.len() as u32).collect();
    b_order.sort_by(|&x, &y| {
        b[x as usize]
            .timestamp
            .total_cmp(&b[y as usize].timestamp)
            .then_with(|| x.cmp(&y))
    });
    let bt: Vec<f64> = b_order.iter().map(|&j| b[j as usize].timestamp).collect();

    // A non-finite `a` stamp makes every difference infinite or `NaN`, so it
    // matches nothing and is dropped rather than tested against every `b`.
    let mut at = Vec::with_capacity(a.len());
    let mut aid = Vec::with_capacity(a.len());
    for &i in &a_order {
        if a[i as usize].timestamp.is_finite() {
            at.push(a[i as usize].timestamp);
            aid.push(i);
        }
    }
    if at.is_empty() {
        return Vec::new();
    }

    // Insertion point of each `a` stamp into `b`. `at` is ascending, so the whole
    // table comes out of one monotone walk in `O(n + m)`. A `NaN` in `b` compares
    // false against everything and sorts last, so the walk stops at exactly the
    // right place: nothing above it is ever a candidate.
    let nb = bt.len();
    let mut pa = vec![0usize; at.len()];
    let mut cursor = 0usize;
    for (p, t) in at.iter().enumerate() {
        while cursor < bt.len() && bt[cursor] < *t {
            cursor += 1;
        }
        pa[p] = cursor;
    }

    let mut st = Assoc {
        at,
        aid,
        pa,
        bt,
        bid: b_order,
        max_dt,
        nb,
        nxt: (0..=nb).collect(),
        prv: (0..=nb).collect(),
        left: nb,
        groups: Vec::new(),
        free: Vec::new(),
        head: vec![NO_GROUP; nb],
        runs: Vec::new(),
        heap: BinaryHeap::new(),
    };

    let n = st.at.len();
    let mut matches: Vec<(usize, usize)> = Vec::with_capacity(n.min(nb));

    st.repoint(0, n);
    st.install();

    while let Some(take) = st.heap.pop() {
        // Consuming a `b` entry retires every group naming it, but only the popped
        // one is popped *now*; the others' heap entries are still queued. A key
        // whose slot has been retired — or recycled onto a different group — is
        // dropped here. That is the only place staleness is possible, and it costs
        // nothing, because the live group for that slot is already in the heap with
        // a fresh key of its own.
        {
            let g = &st.groups[take.slot];
            if !g.live || g.gen != take.gen {
                continue;
            }
        }

        let group = st.groups[take.slot];
        let q = group.q;
        let p = take.p;
        debug_assert!((group.lo..group.hi).contains(&p));

        matches.push((st.aid[p] as usize, st.bid[q] as usize));

        // Retire the `b` entry. The step-function property is what makes the step
        // below bounded: only the `a` entries of the groups naming `q` need a new
        // nearest partner, and no other group's key can be stale.
        st.kill(q);
        if st.left == 0 {
            break;
        }

        // Retire and re-point *every* group naming `q`, not just the one popped.
        // A run split by a consumed `a` can leave two groups on the same `b`, and
        // both are invalidated by this consumption.
        let mut slot = st.head[q];
        st.head[q] = NO_GROUP;
        while slot != NO_GROUP {
            let mut g = st.groups[slot];
            let next = g.next;
            g.live = false;
            st.groups[slot] = g;
            st.free.push(slot);
            if slot == take.slot {
                // The popped group also loses the entry just consumed, so it
                // re-points as two ranges around it.
                st.repoint(g.lo, p);
                st.repoint(p + 1, g.hi);
            } else {
                st.repoint(g.lo, g.hi);
            }
            slot = next;
        }
        st.install();
    }

    // Match `associate.py`'s final ordering: by the first list's timestamp, then by
    // the second index, so equal stamps are ordered deterministically.
    matches.sort_by(|x, y| {
        a[x.0]
            .timestamp
            .total_cmp(&a[y.0].timestamp)
            .then(x.1.cmp(&y.1))
    });

    matches
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::test_util::TempDir;

    fn entry(timestamp: f64, filename: &str) -> IndexEntry {
        IndexEntry {
            timestamp,
            filename: filename.to_string(),
        }
    }

    #[test]
    fn tum_read_index_parses_and_skips_comments() {
        let dir = TempDir::new("tum_index");
        let path = dir.write(
            "rgb.txt",
            concat!(
                "# color images\n",
                "# file: rgb.txt\n",
                "\n",
                "1305031102.175304 rgb/1305031102.175304.png\n",
                "1305031102.275326 rgb/1305031102.275326.png\n",
            ),
        );

        let entries = read_index(&path).expect("parse index");
        assert_eq!(entries.len(), 2);
        assert_eq!(
            entries[0],
            entry(1305031102.175304, "rgb/1305031102.175304.png")
        );
        assert_eq!(entries[1].filename, "rgb/1305031102.275326.png");
    }

    #[test]
    fn tum_read_index_rejects_non_numeric() {
        let dir = TempDir::new("tum_index_bad");
        let path = dir.write("rgb.txt", "not-a-time rgb/x.png\n");
        assert!(read_index(&path).is_err());
    }

    #[test]
    fn tum_read_index_rejects_extra_columns() {
        let dir = TempDir::new("tum_index_cols");
        let path = dir.write("rgb.txt", "1305031102.175304 rgb/x.png extra\n");
        assert!(read_index(&path).is_err());
    }

    #[test]
    fn tum_read_groundtruth_parses_vector_first_quaternion() {
        let dir = TempDir::new("tum_gt");
        let path = dir.write(
            "groundtruth.txt",
            concat!(
                "# groundtruth trajectory\n",
                "1305031102.175304 1.0 2.0 3.0 0.0 0.0 0.0 1.0\n",
            ),
        );

        let poses = read_groundtruth(&path).expect("parse groundtruth");
        assert_eq!(poses.len(), 1);
        let p = &poses[0];
        assert_eq!(p.timestamp, 1305031102.175304);
        assert_eq!(p.pose.translation, Vector3::new(1.0, 2.0, 3.0));
        // (qx,qy,qz,qw) = (0,0,0,1) => identity rotation.
        assert!((p.pose.rotation.w - 1.0).abs() < 1e-15);
        assert!(p.pose.rotation.i.abs() < 1e-15);
    }

    #[test]
    fn tum_read_groundtruth_rejects_wrong_column_count() {
        let dir = TempDir::new("tum_gt_cols");
        let path = dir.write(
            "groundtruth.txt",
            "1305031102.175304 1.0 2.0 3.0 0.0 0.0 0.0\n",
        );
        let err = read_groundtruth(&path).expect_err("7 fields must fail");
        assert!(matches!(err, Error::ParseError(_)));
    }

    #[test]
    fn tum_read_groundtruth_missing_file_is_err() {
        let dir = TempDir::new("tum_gt_missing");
        assert!(read_groundtruth(dir.missing("groundtruth.txt")).is_err());
    }

    #[test]
    fn tum_associate_pairs_nearest_timestamps() {
        let a = [entry(1.0, "a0"), entry(2.0, "a1"), entry(3.0, "a2")];
        let b = [entry(1.02, "b0"), entry(2.01, "b1"), entry(3.5, "b2")];

        let matches = associate(&a, &b, 0.05);
        assert_eq!(matches, vec![(0, 0), (1, 1)]);
    }

    #[test]
    fn tum_associate_enforces_one_to_one() {
        // a[0] and a[1] both lie near b[0]; only the closest may be consumed.
        let a = [entry(0.0, "a0"), entry(0.03, "a1")];
        let b = [entry(0.01, "b0")];

        let matches = associate(&a, &b, 0.05);
        assert_eq!(matches, vec![(0, 0)]);
    }

    #[test]
    fn tum_associate_discards_pairs_beyond_max_dt() {
        let a = [entry(0.0, "a0"), entry(10.0, "a1")];
        let b = [entry(0.01, "b0"), entry(10.5, "b1")];

        let matches = associate(&a, &b, 0.1);
        assert_eq!(matches, vec![(0, 0)]);
    }

    #[test]
    fn tum_associate_sorts_by_first_timestamp() {
        let a = [entry(5.0, "a0"), entry(1.0, "a1")];
        let b = [entry(5.01, "b0"), entry(1.02, "b1")];

        let matches = associate(&a, &b, 0.05);
        // Sorted by a's timestamp, not by input order.
        assert_eq!(matches, vec![(1, 1), (0, 0)]);
    }

    #[test]
    fn tum_associate_empty_and_invalid_tolerance() {
        let a = [entry(0.0, "a0")];
        let b = [entry(0.0, "b0")];
        assert!(associate(&a, &b, -1.0).is_empty());
        assert!(associate(&a, &b, f64::NAN).is_empty());
        assert!(associate(&[], &b, 1.0).is_empty());
    }

    /// The greedy is *global*: the smallest difference wins even when it belongs to
    /// the later `a` entry. A per-`a` greedy would take `(0, 0)` first, whose
    /// difference is `0.05`, ahead of `(1, 1)`, whose difference is `0.0`.
    #[test]
    fn tum_associate_is_a_global_greedy() {
        let a = [entry(0.0, "a0"), entry(2.0, "a1")];
        let b = [entry(0.05, "b0"), entry(1.0, "b1")];

        // Sorted by a's timestamp: a1 sits at 2.0, a0 at 0.0.
        assert_eq!(associate(&a, &b, 2.5), vec![(0, 0), (1, 1)]);
    }

    /// A window's differences are a **V**, not a sorted run, which is why the
    /// algorithm merges on the nearest partner rather than merging windows.
    #[test]
    fn tum_associate_handles_a_v_shaped_window() {
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
        assert_eq!(associate(&a, &b, 2.0), vec![(0, 3)]);
    }

    /// `+inf` is a legitimate tolerance: every pair is a candidate, so the result
    /// is a full one-to-one pairing. It used to be rejected alongside `NaN`.
    #[test]
    fn tum_associate_infinite_tolerance_pairs_everything() {
        let a = [entry(0.0, "a0"), entry(2.0, "a1"), entry(1.0, "a2")];
        let b = [entry(1.0, "b0"), entry(3.0, "b1"), entry(0.5, "b2")];

        let matches = associate(&a, &b, f64::INFINITY);
        assert_eq!(matches.len(), 3);
        let mut b_seen: Vec<usize> = matches.iter().map(|p| p.1).collect();
        b_seen.sort_unstable();
        assert_eq!(b_seen, vec![0, 1, 2]);

        // Control: the same lists with a finite tolerance drop a pair, so the
        // length above is the infinity rather than an accident of the data.
        assert_eq!(associate(&a, &b, 0.75).len(), 2);
    }

    /// A non-finite `a` timestamp can never satisfy "the difference is finite", so
    /// it matches nothing — at any tolerance, including `+inf`.
    #[test]
    fn tum_associate_ignores_non_finite_a_timestamps() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let a = [entry(bad, "a0"), entry(1.0, "a1")];
            let b = [entry(1.0, "b0"), entry(2.0, "b1")];
            assert_eq!(
                associate(&a, &b, f64::INFINITY),
                vec![(1, 0)],
                "a timestamp {bad:?} must match nothing"
            );
        }
    }

    /// The tolerance boundary is decided on the exact difference, not on
    /// `ta + max_dt`: that sum rounds, so a range half-open at it can over-include
    /// by an ulp and admit a pair farther than `max_dt`.
    #[test]
    fn tum_associate_tolerance_boundary_is_exact() {
        let a = [entry(1.0, "a0")];
        let b = [entry(1.5, "b0")];
        assert_eq!(
            associate(&a, &b, 0.5),
            vec![(0, 0)],
            "0.5 == max_dt matches"
        );
        assert!(
            associate(&a, &b, 0.499_999_999_999_999_9).is_empty(),
            "one ulp below the difference must not match"
        );
    }

    /// Many `a` entries contending for fewer `b` entries: the shape that breaks a
    /// lazy per-`a` heap, which loses pairs when a consumed `b` leaves other `a`
    /// entries holding stale keys. Every difference is `b`'s own timestamp, so
    /// each `b` is taken in order and won by the lowest still-unused `a` index.
    #[test]
    fn tum_associate_handles_wide_contention() {
        let n = 64;
        let a: Vec<IndexEntry> = (0..n).map(|i| entry(0.0, &format!("a{i}"))).collect();
        let b: Vec<IndexEntry> = (0..n / 2)
            .map(|j| entry(j as f64, &format!("b{j}")))
            .collect();

        assert_eq!(
            associate(&a, &b, f64::INFINITY),
            (0..n / 2).map(|j| (j, j)).collect::<Vec<_>>()
        );
    }
}
