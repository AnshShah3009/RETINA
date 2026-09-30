#![allow(dead_code)]
//! Shared scaffolding for the `cv-io` parser robustness tests.
//!
//! The parsers in `crates/io/src/` take untrusted binary input, so every test
//! in this suite has to be able to distinguish three outcomes that a plain
//! `assert!(read_x(bytes).is_ok())` conflates:
//!
//! 1. the parser returned a value (possibly a *wrong* one),
//! 2. the parser returned an error,
//! 3. the parser **panicked** or **hung** - neither of which is acceptable, and
//!    neither of which `cargo test` can report as an ordinary failure without
//!    help, because a panic kills the whole test binary and a hang kills CI.
//!
//! [`run_with_budget`] runs the parser on a dedicated thread and gives up after
//! a wall-clock budget. A runaway thread is deliberately *not* joined: joining
//! would block the test forever, which is exactly the failure being detected.
//!
//! The malformed bytes are always built here or in the individual test file -
//! no fixture files on disk - so each test is self-contained and reviewable.

use std::any::Any;
use std::fs;
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Default wall-clock budget for a single parser call.
pub const DEFAULT_BUDGET: Duration = Duration::from_secs(20);
/// Budget for the tests that deliberately provoke hostile header counts.
pub const HOSTILE_BUDGET: Duration = Duration::from_secs(60);

/// What happened when a parser was fed a byte string.
pub enum Outcome<T> {
    /// The parser returned `Ok`.
    Returned(T),
    /// The parser returned `Err` - an acceptable outcome.
    Error(String),
    /// The parser panicked. Always a bug: a library must not panic on input.
    Panicked(String),
    /// The parser did not finish inside the budget. Always a bug.
    TimedOut,
}

impl<T> Outcome<T> {
    /// A short human-readable tag, used in assertion messages.
    pub fn kind(&self) -> &'static str {
        match self {
            Outcome::Returned(_) => "Ok",
            Outcome::Error(_) => "Err",
            Outcome::Panicked(_) => "PANIC",
            Outcome::TimedOut => "HANG",
        }
    }

    /// True for the two acceptable outcomes: a value or an error.
    pub fn is_ok_or_err(&self) -> bool {
        matches!(self, Outcome::Returned(_) | Outcome::Error(_))
    }

    /// Consume the outcome, requiring `Err`, and return its message.
    pub fn expect_err(self, what: &str) -> String {
        match self {
            Outcome::Error(msg) => msg,
            Outcome::Returned(_) => panic!("{what}: expected Err, got Ok"),
            Outcome::Panicked(msg) => panic!("{what}: expected Err, got a PANIC: {msg}"),
            Outcome::TimedOut => panic!("{what}: expected Err, got a HANG"),
        }
    }

    /// Consume the outcome, requiring `Ok`, and return the value.
    pub fn expect_ok(self, what: &str) -> T {
        match self {
            Outcome::Returned(v) => v,
            Outcome::Error(msg) => panic!("{what}: expected Ok, got Err({msg})"),
            Outcome::Panicked(msg) => panic!("{what}: expected Ok, got a PANIC: {msg}"),
            Outcome::TimedOut => panic!("{what}: expected Ok, got a HANG"),
        }
    }
}

/// Run `f` on a dedicated thread, converting a panic into [`Outcome::Panicked`]
/// and a wall-clock overrun into [`Outcome::TimedOut`].
pub fn run_with_budget<T, F>(budget: Duration, f: F) -> Outcome<T>
where
    F: FnOnce() -> cv_core::Result<T> + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = mpsc::channel::<Outcome<T>>();

    let handle = thread::Builder::new()
        .name("cv-io-parser-under-test".to_string())
        .stack_size(4 * 1024 * 1024)
        .spawn(move || {
            let result = std::panic::catch_unwind(AssertUnwindSafe(f));
            let outcome = match result {
                Ok(Ok(v)) => Outcome::Returned(v),
                Ok(Err(e)) => Outcome::Error(format!("{e}")),
                Err(payload) => Outcome::Panicked(panic_message(&payload)),
            };
            let _ = tx.send(outcome);
        })
        .expect("spawn parser thread");

    match rx.recv_timeout(budget) {
        Ok(outcome) => {
            let _ = handle.join();
            outcome
        }
        Err(_) => {
            // Never join: the thread is still spinning inside the parser, and
            // joining it would hang the test suite forever.
            drop(handle);
            Outcome::TimedOut
        }
    }
}

/// Same as [`run_with_budget`] with [`DEFAULT_BUDGET`].
pub fn run<T, F>(f: F) -> Outcome<T>
where
    F: FnOnce() -> cv_core::Result<T> + Send + 'static,
    T: Send + 'static,
{
    run_with_budget(DEFAULT_BUDGET, f)
}

// ---------------------------------------------------------------------------
// Parser runners
// ---------------------------------------------------------------------------

/// A runner takes the bytes to parse and returns the result. Using a `fn` (not
/// a closure) keeps the call sites free of `'static` borrow gymnastics, and
/// makes it impossible to accidentally assert on state that the parser mutated.
pub type Runner = fn(Vec<u8>) -> cv_core::Result<Vec<nalgebra::Point3<f32>>>;

macro_rules! point_cloud_runner {
    ($name:ident, $path:ident, $f:path) => {
        /// `BufRead` is required, so the bytes go through a `BufReader`.
        pub fn $name(bytes: Vec<u8>) -> cv_core::Result<Vec<nalgebra::Point3<f32>>> {
            $f(std::io::BufReader::new(std::io::Cursor::new(bytes))).map(|c| c.points)
        }
    };
}

point_cloud_runner!(run_pcd, pcd, cv_io::pcd::read_pcd);
point_cloud_runner!(run_ply, ply, cv_io::ply::read_ply);
point_cloud_runner!(run_obj, obj, cv_io::obj::read_obj);

/// `read_stl` returns a mesh; the runner projects it to its vertex list so the
/// same `Runner` signature covers every parser.
pub fn run_stl(bytes: Vec<u8>) -> cv_core::Result<Vec<nalgebra::Point3<f32>>> {
    cv_io::stl::read_stl(std::io::Cursor::new(bytes)).map(|m| m.vertices)
}

/// Run a parser over `bytes` with the default budget.
pub fn parse(runner: Runner, bytes: &[u8]) -> Outcome<Vec<nalgebra::Point3<f32>>> {
    let bytes = bytes.to_vec();
    run_with_budget(DEFAULT_BUDGET, move || runner(bytes))
}

/// Run a parser over `bytes` with the longer [`HOSTILE_BUDGET`].
pub fn parse_hostile(runner: Runner, bytes: &[u8]) -> Outcome<Vec<nalgebra::Point3<f32>>> {
    let bytes = bytes.to_vec();
    run_with_budget(HOSTILE_BUDGET, move || runner(bytes))
}

/// [`parse`] for a file that is expected to fail: never panics, never hangs.
pub fn parse_err(runner: Runner, bytes: &[u8], what: &str) -> String {
    parse(runner, bytes).expect_err(what)
}

/// [`parse_hostile`] for a file that is expected to fail.
pub fn parse_hostile_err(runner: Runner, bytes: &[u8], what: &str) -> String {
    parse_hostile(runner, bytes).expect_err(what)
}

/// Extract a readable message from a panic payload.
fn panic_message(payload: &Box<dyn Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

// ---------------------------------------------------------------------------
// Known-bug catalogue
// ---------------------------------------------------------------------------

/// A bug that has been reproduced and is therefore *expected* to be observed by
/// the deterministic tests and by the mutation corpus.
///
/// The catalogue keeps the suite honest in both directions: a fixed bug that
/// comes back fails loudly, and the fuzz corpus fails on any panic nobody has
/// looked at yet.
pub struct KnownBug {
    /// `pcd`, `ply`, `stl`, `obj`, ...
    pub parser: &'static str,
    /// What the input is.
    pub what: &'static str,
    /// A substring of the observed symptom.
    pub signature: &'static str,
    /// Minimal reproduction.
    pub repro: &'static str,
}

pub const KNOWN_BUGS: &[KnownBug] = &[
    KnownBug {
        parser: "pcd",
        what: "WIDTH*HEIGHT overflows usize (no POINTS line)",
        signature: "multiply with overflow",
        repro: "WIDTH 18446744073709551615 / HEIGHT 2 / DATA binary, no body",
    },
    KnownBug {
        parser: "ply",
        what: "body after a second element is read as vertex data",
        signature: "reads the first element line only",
        repro: "element vertex 1 / element face 1 with a 2-line body",
    },
    KnownBug {
        parser: "ply",
        what: "NaN/Inf coordinates parse as valid f32",
        signature: "non-finite",
        repro: "element vertex 1 + body `nan inf -inf`",
    },
    KnownBug {
        parser: "obj",
        what: "out-of-range face indices are accepted",
        signature: "out-of-range",
        repro: "`f 1 2 999999` with 2 vertices",
    },
    KnownBug {
        parser: "obj",
        what: "NaN/Inf coordinates parse as valid f32",
        signature: "non-finite",
        repro: "`v nan inf -inf`",
    },
    KnownBug {
        parser: "stl",
        what: "a truncated ASCII STL returns Ok with zero faces",
        signature: "a truncated ASCII STL",
        repro: "header + 3 `vertex` lines, no `endloop`",
    },
    KnownBug {
        parser: "stl",
        what: "a vertex line with too few fields is skipped silently",
        signature: "missing z",
        repro: "`vertex 1 2` (missing z)",
    },
    KnownBug {
        parser: "stl",
        what: "NaN/Inf vertices are accepted",
        signature: "non-finite",
        repro: "`vertex nan inf -inf`",
    },
    KnownBug {
        parser: "kitti",
        what: "a non-orthonormal pose is accepted and silently re-orthonormalised",
        signature: "scale-2 'rotation'",
        repro: "poses.txt with a 2*I rotation block",
    },
];

/// Does this observed failure match one of the already-known bugs?
pub fn known_bug_for(parser: &str, signature: &str) -> Option<&'static KnownBug> {
    KNOWN_BUGS
        .iter()
        .find(|b| b.parser == parser && signature.contains(b.signature))
}

// ---------------------------------------------------------------------------
// Temp files
// ---------------------------------------------------------------------------

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// A self-deleting temporary directory (the crate deliberately has no
/// `tempfile` dependency).
pub struct TempDir {
    path: PathBuf,
}

impl TempDir {
    pub fn new(tag: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut path = std::env::temp_dir();
        path.push(format!("cv_io_robust_{tag}_{}_{seq}_{nanos}", std::process::id()));
        fs::create_dir_all(&path).expect("create temp dir");
        Self { path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Write raw bytes to `name` and return the path.
    pub fn write_bytes(&self, name: &str, contents: &[u8]) -> PathBuf {
        let p = self.path.join(name);
        fs::write(&p, contents).expect("write temp file");
        p
    }

    /// Write text to `name` and return the path.
    pub fn write(&self, name: &str, contents: &str) -> PathBuf {
        self.write_bytes(name, contents.as_bytes())
    }

    /// A path inside the directory that does not exist.
    pub fn missing(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

// ---------------------------------------------------------------------------
// Address-space probe (Linux only)
// ---------------------------------------------------------------------------

/// Current virtual address space of the process in bytes, or `None` where
/// `/proc` is unavailable.
///
/// Used to assert that a hostile header count does not commit tens of gigabytes
/// of address space before a single point is read.
#[cfg(target_os = "linux")]
pub fn address_space_bytes() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmSize:") {
            let kb: u64 = rest.trim().trim_end_matches(" kB").trim().parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
pub fn address_space_bytes() -> Option<u64> {
    None
}

// ---------------------------------------------------------------------------
// Byte builders
// ---------------------------------------------------------------------------

/// Concatenate byte chunks.
pub fn bytes(parts: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    for p in parts {
        out.extend_from_slice(p);
    }
    out
}

/// f32 values as little-endian bytes.
pub fn f32s(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A 50-byte binary STL triangle record: normal, three vertices, 2 attribute
/// bytes.
pub fn stl_triangle(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> Vec<u8> {
    let mut t = Vec::with_capacity(50);
    t.extend_from_slice(&f32s(&[0.0, 0.0, 1.0]));
    t.extend_from_slice(&f32s(&a));
    t.extend_from_slice(&f32s(&b));
    t.extend_from_slice(&f32s(&c));
    t.extend_from_slice(&[0, 0]);
    t
}

/// A binary STL file with an 80-byte header holding `header_text` (zero padded)
/// and a little-endian `tri_count`.
pub fn stl_binary(header_text: &str, tri_count: u32, body: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; 80];
    for (i, b) in header_text.bytes().take(80).enumerate() {
        out[i] = b;
    }
    out.extend_from_slice(&tri_count.to_le_bytes());
    out.extend_from_slice(body);
    out
}

/// A binary PCD file built from a caller-supplied `WIDTH`/`POINTS` value plus an
/// independent body.
///
/// Keeping the header count and the body length independent is the whole point:
/// that disagreement is what these tests are about.
pub fn pcd_binary(points: &str, header_extra: &str, body: &[u8]) -> Vec<u8> {
    let head = format!(
        "# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n\
         WIDTH {points}\nHEIGHT 1\n{header_extra}DATA binary\n"
    );
    bytes(&[head.as_bytes(), body])
}

/// An ASCII PCD file with the `x y z` float header.
pub fn pcd_ascii(points: &str, header_extra: &str, body: &str) -> Vec<u8> {
    let head = format!(
        "# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n\
         WIDTH {points}\nHEIGHT 1\n{header_extra}DATA ascii\n"
    );
    bytes(&[head.as_bytes(), body.as_bytes()])
}

/// A binary_compressed PCD file: header, the two u32 size words, then
/// `compressed`.
pub fn pcd_binary_compressed(points: &str, compressed: &[u8], uncompressed_size: u32) -> Vec<u8> {
    let head = format!(
        "# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n\
         WIDTH {points}\nHEIGHT 1\nPOINTS {points}\nDATA binary_compressed\n"
    );
    let mut out = head.into_bytes();
    out.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
    out.extend_from_slice(&uncompressed_size.to_le_bytes());
    out.extend_from_slice(compressed);
    out
}

/// A minimal, valid ASCII PLY with `n` vertices and no optional attributes.
pub fn ply_ascii(n: usize) -> Vec<u8> {
    let mut s = String::from("ply\nformat ascii 1.0\nelement vertex ");
    s.push_str(&n.to_string());
    s.push_str("\nproperty float x\nproperty float y\nproperty float z\nend_header\n");
    for i in 0..n {
        s.push_str(&format!("{i} {} 0\n", i as f32 + 0.5));
    }
    s.into_bytes()
}

/// A minimal, valid ASCII STL with one facet.
pub fn stl_ascii() -> Vec<u8> {
    b"solid x\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid x\n".to_vec()
}

/// A minimal, valid OBJ with three vertices and one face.
pub fn obj_ascii() -> Vec<u8> {
    b"# comment\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n".to_vec()
}

// ---------------------------------------------------------------------------
// Assertions shared by the suites
// ---------------------------------------------------------------------------

/// Assert a point list contains no non-finite coordinate.
///
/// A NaN that reaches a returned point cloud is a *plausible wrong result*:
/// every downstream consumer (ICP, RANSAC, voxel hashing) silently produces
/// garbage from it, and nothing in the returned type says "this is NaN".
pub fn assert_all_finite(points: &[nalgebra::Point3<f32>], what: &str) {
    for (i, p) in points.iter().enumerate() {
        assert!(
            p.x.is_finite() && p.y.is_finite() && p.z.is_finite(),
            "{what}: point {i} is not finite: ({}, {}, {})",
            p.x,
            p.y,
            p.z
        );
    }
}

/// Lowercase hex dump, for bug reports.
#[allow(dead_code)]
pub fn hex(data: &[u8]) -> String {
    data.iter()
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// An escaped, printable rendering of a byte string, for bug reports.
#[allow(dead_code)]
pub fn escaped(data: &[u8]) -> String {
    let mut s = String::new();
    for &b in data {
        match b {
            b'\n' => s.push_str("\\n"),
            b'\r' => s.push_str("\\r"),
            b'\t' => s.push_str("\\t"),
            0x20..=0x7e => s.push(b as char),
            other => s.push_str(&format!("\\x{other:02x}")),
        }
    }
    s
}
