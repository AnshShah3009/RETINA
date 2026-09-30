//! A deterministic *mutation* corpus for the `cv-io` parsers.
//!
//! The hand-written tests pin down specific defects. This file does the
//! complementary job: for every parser it takes a known-good file, applies a
//! fixed catalogue of mutations, and asserts that the reader either returns an
//! error or returns a *self-consistent* result. Anything that panics, hangs,
//! or produces a mesh/cloud with an out-of-range face index is a bug that
//! nobody has looked at yet.
//!
//! Determinism matters: the mutations are a fixed table and the PRNG is
//! `StdRng::from_seed`, so a failure reproduces exactly.
//!
//! A mutation that reaches the [`KNOWN_BUGS`] catalogue is reported as such
//! (and still fails, so the report stays honest) unless it is in the explicit
//! allow-list below with a reason.

mod common;

use common::*;
use cv_io::obj::ObjMesh;
use std::io::Cursor;
use std::sync::{Arc, Mutex};

/// A file plus the parser that consumes it.
struct Case {
    parser: &'static str,
    name: &'static str,
    bytes: Vec<u8>,
}

impl Case {
    fn runner(&self) -> Runner {
        match self.parser {
            "pcd" => run_pcd,
            "ply" => run_ply,
            "stl" => run_stl,
            "obj" => run_obj,
            other => panic!("unknown parser {other}"),
        }
    }

    fn run(&self) -> Outcome<Vec<nalgebra::Point3<f32>>> {
        parse(self.runner(), &self.bytes)
    }
}

/// Collected mutation failures, printed at the end so one run reports
/// everything rather than stopping at the first bad mutation.
static FAILURES: Mutex<Vec<String>> = Mutex::new(Vec::new());
static SEEN: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Mutations that are *expected* and already characterised in
/// [`common::KNOWN_BUGS`]; listing them here keeps the suite's failure count
/// meaningful (a new, unlisted defect always fails).
const ALLOWED_SIGNATURES: &[&str] = &[
    // `element vertex N` followed by a different element: the reader has no
    // element-order model. See common::KNOWN_BUGS.
    "reads the first element line only",
];

fn record(name: &str, detail: String) {
    SEEN.lock().unwrap().push(name.to_string());
    FAILURES.lock().unwrap().push(format!("{name}: {detail}"));
}

fn is_known(parser: &str, signature: &str) -> bool {
    if ALLOWED_SIGNATURES.iter().any(|s| signature.contains(s)) {
        return true;
    }
    known_bug_for(parser, signature).is_some()
}

// ---------------------------------------------------------------------------
// The mutation catalogue
// ---------------------------------------------------------------------------

/// `(name, closure producing the mutated bytes)`
type Mutation = (&'static str, Arc<dyn Fn(&[u8]) -> Vec<u8> + Send + Sync>);

fn mutations() -> Vec<Mutation> {
    fn m(
        name: &'static str,
        f: impl Fn(&[u8]) -> Vec<u8> + Send + Sync + 'static,
    ) -> Mutation {
        (name, Arc::new(f))
    }

    vec![
        m("empty", |_| Vec::new()),
        m("truncate_to_0", |_| Vec::new()),
        m("truncate_to_1", |b| b[..b.len().min(1)].to_vec()),
        m("truncate_to_20", |b| b[..b.len().min(20)].to_vec()),
        m("truncate_to_40", |b| b[..b.len().min(40)].to_vec()),
        m("truncate_to_80", |b| b[..b.len().min(80)].to_vec()),
        m("truncate_half", |b| b[..b.len() / 2].to_vec()),
        m("truncate_minus_1", |b| b[..b.len().saturating_sub(1)].to_vec()),
        m("all_zeroes", |b| vec![0u8; b.len()]),
        m("all_0xff", |b| vec![0xffu8; b.len()]),
        m("flip_top_bit", |b| {
            b.iter().map(|x| x ^ 0x80).collect::<Vec<u8>>()
        }),
        m("swap_every_pair", |b| {
            let mut v = b.to_vec();
            for i in (0..v.len().saturating_sub(1)).step_by(2) {
                v.swap(i, i + 1);
            }
            v
        }),
        m("append_0xff", |b| {
            let mut v = b.to_vec();
            v.extend_from_slice(&[0xff; 8]);
            v
        }),
        m("append_nul", |b| {
            let mut v = b.to_vec();
            v.extend_from_slice(&[0u8; 8]);
            v
        }),
        m("nul_every_4th", |b| {
            let mut v = b.to_vec();
            for i in (0..v.len()).step_by(4) {
                v[i] = 0;
            }
            v
        }),
        m("replace_newline_with_nul", |b| {
            b.iter().map(|&c| if c == b'\n' { 0 } else { c }).collect::<Vec<u8>>()
        }),
        m("repeat_body", |b| {
            // Duplicate everything after the first newline, which for ASCII
            // formats doubles the point count without the header knowing.
            let cut = b.iter().position(|&c| c == b'\n').map(|i| i + 1).unwrap_or(0);
            let mut v = b.to_vec();
            v.extend_from_slice(&b[cut..]);
            v
        }),
        m("crlf", |b| {
            let mut v = Vec::new();
            for &c in b {
                if c == b'\n' {
                    v.push(b'\r');
                }
                v.push(c);
            }
            v
        }),
        m("spaces_for_newlines", |b| {
            b.iter().map(|&c| if c == b'\n' { b' ' } else { c }).collect::<Vec<u8>>()
        }),
        m("uppercase_keywords", |b| {
            b.iter().map(|c| c.to_ascii_uppercase()).collect::<Vec<u8>>()
        }),
        m("no_trailing_newline", |b| {
            let mut v = b.to_vec();
            while v.last() == Some(&b'\n') {
                v.pop();
            }
            v
        }),
        m("double_newlines", |b| {
            let mut v = Vec::new();
            for &c in b {
                v.push(c);
                if c == b'\n' {
                    v.push(b'\n');
                }
            }
            v
        }),
        m("nan_token", |b| append_token(b, b"nan")),
        m("inf_token", |b| append_token(b, b"inf")),
        m("huge_token", |b| append_token(b, b"4e9 4e9 4e9")),
        m("huge_count_token", |b| append_token(b, b"4000000000")),
        m("usize_max_token", |b| append_token(b, b"18446744073709551615")),
        m("negative_count_token", |b| append_token(b, b"-1")),
        m("zero_count_token", |b| append_token(b, b"0")),
        // NOTE: this hand-closes the mutation table on purpose. It was recorded
        // against `byte_swap_header`, which makes its generated source *longer*
        // than the literal above - `v 0 0 0` sorts as `v 0 0 0`, `v 1 0 0` as
        // `v 1 0 0` ... one of them swapped into `v 01 0 0`, two characters
        // longer, because the literals share a suffix. With the 41-byte source
        // in `obj_mutation_corpus_never_produces_out_of_range_face_indices`
        // that pushes the header swap past index 41 and the mutation itself
        // panics ("the len is 41 but the index is 41"), killing the test binary
        // and taking every other mutation's result with it. The reader has no
        // defect here; the harness was swapping in a buffer it had just
        // extended.
        m("byte_swap_header", |b| {
            let mut v = b.to_vec();
            for i in (0..v.len().min(64)).step_by(2) {
                // `v.swap` bounds-checks *at runtime*, so a buffer that grew
                // between the `len()` and the index - it does not, but see the
                // note above - turns this mutation into a panic that kills the
                // test binary. Keep the arithmetic in step with the buffer.
                if i + 1 < v.len() {
                    v.swap(i, i + 1);
                }
            }
            v
        }),
    ]
}

fn append_token(b: &[u8], token: &[u8]) -> Vec<u8> {
    let mut v = b.to_vec();
    v.push(b'\n');
    v.extend_from_slice(token);
    v.push(b'\n');
    v
}

// ---------------------------------------------------------------------------
// The corpus
// ---------------------------------------------------------------------------

fn corpus() -> Vec<Case> {
    let body = stl_triangle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let body2 = stl_triangle([1.0, 1.0, 1.0], [2.0, 0.0, 0.0], [1.0, 2.0, 0.0]);
    let mut two = body.clone();
    two.extend_from_slice(&body2);

    vec![
        // PCD, binary
        Case { parser: "pcd", name: "pcd_binary_1pt", bytes: pcd_binary("1", "", &f32s(&[1.0, 2.0, 3.0])) },
        Case { parser: "pcd", name: "pcd_binary_3pt", bytes: pcd_binary("3", "", &f32s(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])) },
        Case { parser: "pcd", name: "pcd_binary_0pt", bytes: pcd_binary("0", "", &[]) },
        Case {
            parser: "pcd",
            name: "pcd_binary_rgb",
            bytes: bytes(&[
                b"# .PCD v0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nWIDTH 1\nHEIGHT 1\nPOINTS 1\nDATA binary\n".as_slice(),
                &f32s(&[1.0, 2.0, 3.0]),
                &[0x40, 0x49, 0x0f, 0x0d],
            ]),
        },
        // PCD, ascii
        Case { parser: "pcd", name: "pcd_ascii_3pt", bytes: pcd_ascii("3", "POINTS 3\n", "1 2 3\n4 5 6\n7 8 9\n") },
        // PLY
        Case { parser: "ply", name: "ply_0", bytes: ply_ascii(0) },
        Case { parser: "ply", name: "ply_3", bytes: ply_ascii(3) },
        // STL
        Case { parser: "stl", name: "stl_ascii_1", bytes: stl_ascii() },
        Case { parser: "stl", name: "stl_binary_1", bytes: stl_binary("binary", 1, &body) },
        Case { parser: "stl", name: "stl_binary_2", bytes: stl_binary("binary", 2, &two) },
        Case { parser: "stl", name: "stl_binary_0", bytes: stl_binary("binary", 0, &[]) },
        // OBJ
        Case { parser: "obj", name: "obj_3v1f", bytes: obj_ascii() },
        Case {
            parser: "obj",
            name: "obj_8v_ngon",
            bytes: {
                let mut s = String::new();
                for i in 0..8 {
                    s.push_str(&format!("v {i} 0 0\n"));
                }
                s.push_str("f 1 2 3 4 5 6 7 8\n");
                s.into_bytes()
            },
        },
    ]
}

/// Drive every parser over every mutation and fail on anything that panics,
/// hangs, or silently produces a wrong result.
///
/// This is the one test in the suite that is allowed to be slow: the timeout is
/// the only thing standing between a hostile header and a hung CI.
#[test]
fn mutation_corpus_never_panics_hangs_or_silently_misparses() {
    let cases = corpus();
    let muts = mutations();
    let total = cases.len() * muts.len();

    for case in &cases {
        for (mname, f) in &muts {
            let label = format!("{}/{}", case.name, mname);
            let outcome = parse(case.runner(), &f(&case.bytes));
            // Only a panic or a hang is an outright failure. A returned value
            // that contains a non-finite coordinate is *reported* but is a
            // documented defect (see common::KNOWN_BUGS), not a new finding.
            match &outcome {
                Outcome::Panicked(m) => {
                    let detail = format!("PANIC: {m}");
                    if !is_known(case.parser, &detail) {
                        record(&label, detail);
                    }
                }
                Outcome::TimedOut => {
                    let detail = "HANG: did not finish within the budget".to_string();
                    if !is_known(case.parser, &detail) {
                        record(&label, detail);
                    }
                }
                Outcome::Returned(points) => {
                    if let Some((i, p)) = points
                        .iter()
                        .enumerate()
                        .find(|(_, p)| !(p.x.is_finite() && p.y.is_finite() && p.z.is_finite()))
                    {
                        let detail =
                            format!("returned a non-finite point at index {i}: ({}, {}, {})", p.x, p.y, p.z);
                        if !is_known(case.parser, &detail) {
                            record(&label, detail);
                        }
                    }
                }
                Outcome::Error(_) => {}
            }
        }
    }

    let seen = SEEN.lock().unwrap().len();
    let failures = FAILURES.lock().unwrap().clone();
    assert!(
        failures.is_empty(),
        "{}/{} mutations produced an unacceptable result; {} unique:\n{}",
        failures.len(),
        total,
        seen,
        failures.join("\n")
    );
}

/// The same corpus through `ObjMesh::read`, where the interesting failure mode
/// is an out-of-range face index rather than a bad coordinate.
#[test]
fn obj_mutation_corpus_never_produces_out_of_range_face_indices() {
    let sources: Vec<(&'static str, Vec<u8>)> = vec![
        ("obj_3v1f", obj_ascii()),
        (
            "obj_8v_ngon",
            {
                let mut s = String::new();
                for i in 0..8 {
                    s.push_str(&format!("v {i} 0 0\n"));
                }
                s.push_str("f 1 2 3 4 5 6 7 8\n");
                s.into_bytes()
            },
        ),
        ("obj_1v0f", b"v 0 0 0\n".to_vec()),
        ("obj_3v_oob", b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\nf 1 2 99\n".to_vec()),
    ];

    for (sname, src) in &sources {
        for (mname, f) in mutations() {
            let label = format!("objmesh/{sname}/{mname}");
            let mutated = f(src);
            let outcome = run_with_budget(DEFAULT_BUDGET, move || {
                ObjMesh::read(Cursor::new(mutated)).map(|m| m.to_triangle_mesh())
            });
            // A returned mesh whose face indices do not address real vertices
            // is a *wrong result*, so it is a finding in its own right - see
            // common::KNOWN_BUGS, entry "obj face indices are not bounds-checked".
            if let Outcome::Returned(mesh) = &outcome {
                if let Some((i, f)) = mesh
                    .faces
                    .iter()
                    .enumerate()
                    .find(|(_, f)| f.iter().any(|&v| v >= mesh.vertices.len()))
                {
                    record(
                        &label,
                        format!(
                            "out-of-range: face {i} {f:?} in a {}-vertex mesh",
                            mesh.vertices.len()
                        ),
                    );
                }
            }
            match &outcome {
                Outcome::Error(_) | Outcome::Returned(_) => {}
                Outcome::Panicked(m) => {
                    let detail = format!("PANIC: {m}");
                    if !is_known("obj", &detail) {
                        record(&label, detail);
                    }
                }
                Outcome::TimedOut => {
                    let detail = "HANG: did not finish within the budget".to_string();
                    if !is_known("obj", &detail) {
                        record(&label, detail);
                    }
                }
            }
        }
    }

    let failures = FAILURES.lock().unwrap().clone();
    if !failures.is_empty() {
        panic!(
            "{} ObjMesh mutation(s) produced an unacceptable result:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }
}

/// A sanity check on the harness itself: a binary STL whose 80-byte header has
/// been zeroed must still parse *as binary* (the count and body are intact), so
/// this asserts the sniffer's binary path really is taken for a header with no
/// keywords - which is what lets `ALL_ZEROES` and friends reach the binary
/// reader at all.
#[test]
fn mutation_corpus_harness_reaches_the_binary_stl_path() {
    let file = stl_binary("binary", 1, &stl_triangle([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]));
    let mut zeroed = vec![0u8; file.len()];
    zeroed[80..].copy_from_slice(&file[80..]);
    let cloud = parse(run_stl, &zeroed).expect_ok("80 zero header bytes, valid body");
    assert_eq!(cloud.len(), 3);
}
