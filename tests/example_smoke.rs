// Every example is compiled by CI and executed by nothing.
//
// This module was written after `orb_benchmark` was found to panic on its first
// line: `Orb::detect_and_compute_ctx` had never run in any process, because
// nothing called it, and the GPU parity suite reaches `fast_detect` through a
// different entry point. An example that builds is not an example that runs.
//
// Each case below runs the built binary and asserts it exits cleanly. The list is
// explicit rather than discovered, so an example that needs arguments or a
// dataset is not silently skipped - adding one here is a deliberate act.
//
// Run with:
//   cargo test -p rust-cv-native --test example_smoke -- --nocapture
//
// Note this requires the examples to have been built first:
//   cargo build --release --workspace --examples

use std::path::PathBuf;
use std::process::Command;

/// Seconds a smoke example may take. Generous - a few of these initialise a GPU
/// context, which is slow on a cold driver - but short enough that a hang is
/// reported rather than waited on.
const TIMEOUT_SECS: u64 = 120;

/// Locate a built example binary, preferring release and falling back to debug.
fn example_path(name: &str) -> Option<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for profile in ["release", "debug"] {
        let p = manifest
            .join("target")
            .join(profile)
            .join("examples")
            .join(name);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn run(name: &str) {
    let Some(path) = example_path(name) else {
        eprintln!("skipping {name}: not built (cargo build --workspace --examples)");
        return;
    };

    let mut child = match Command::new(&path)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => panic!("{name}: could not be spawned: {e}"),
    };

    // Poll rather than `wait_with_output`, so a hang is a failure with a name
    // rather than a stalled test run.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(TIMEOUT_SECS);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let out = child.wait_with_output().ok();
                let stderr = out
                    .map(|o| String::from_utf8_lossy(&o.stderr).to_string())
                    .unwrap_or_default();
                assert!(
                    status.success(),
                    "{name} exited {status}\n--- stderr ---\n{}",
                    stderr.lines().rev().take(12).collect::<Vec<_>>().join("\n")
                );
                return;
            }
            Ok(None) => {
                if std::time::Instant::now() > deadline {
                    let _ = child.kill();
                    panic!("{name} did not finish within {TIMEOUT_SECS}s");
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            Err(e) => panic!("{name}: wait failed: {e}"),
        }
    }
}

macro_rules! smoke {
    ($($name:ident),* $(,)?) => {
        $(
            #[test]
            fn $name() {
                run(stringify!($name));
            }
        )*
    };
}

smoke!(
    plot_demo,
    registration_icp,
    gaussian_splatting_basic,
    kalman_filter,
    bundle_adjustment,
    raycasting,
    scientific,
    core_types,
    imgproc_demo,
    features_demo,
    orb_benchmark,
);

/// The viewer demo opens a window and runs until it is closed, so it can never
/// be "finish within 120s". It is spawned, given a moment to fail, and then
/// killed - asserting only that it did not die on startup.
///
/// This is the third time this example has broken the smoke suite, and the
/// first two were the viewer's own doing: a launcher that printed to stdout, and
/// then one that rendered. A GUI example and an "every example exits" harness are
/// incompatible, so the harness learned about them rather than the examples
/// pretending to be batch jobs.
#[test]
fn viewer_demo_opens_and_survives_startup() {
    let Some(path) = example_path("demo") else {
        eprintln!("skipping demo: not built");
        return;
    };
    let mut child = Command::new(&path)
        .spawn()
        .expect("demo should be spawnable");

    // Long enough to get past pipeline creation, which is where the renderer
    // reports a missing device or a shader that will not compile.
    std::thread::sleep(std::time::Duration::from_secs(5));

    match child.try_wait() {
        // Still running is the success case: the window is up.
        Ok(None) => {
            let _ = child.kill();
        }
        Ok(Some(status)) => panic!(
            "demo exited {status} instead of staying open - \
             it is a GUI app and should keep running"
        ),
        Err(e) => {
            let _ = child.kill();
            panic!("demo: wait failed: {e}");
        }
    }
}

/// `orbdiag` is a long-running diagnostic that sweeps feature counts and runs
/// the mapper on each, so it is exercised by one size rather than the whole
/// sweep - it registered 20/20 at every size when checked, and running it in
/// full takes longer than a smoke suite should.
#[test]
fn orbdiag_starts_and_reports() {
    let Some(path) = example_path("orbdiag") else {
        eprintln!("skipping orbdiag: not built");
        return;
    };
    // `.output()` waits for the process to exit, and orbdiag sweeps feature
    // counts over the full dataset - it runs for many minutes and never
    // returns. The test therefore blocked on it and could only ever pass if the
    // example happened to finish first; on CI it did not, and the assertion
    // fired on an empty stdout rather than on anything real.
    //
    // The pipe is drained on a background thread while the main thread polls
    // for the report, with a deadline. Joining the reader directly is not
    // enough: it returns as soon as the child closes the pipe, which on CI
    // happened before any report was written.
    let mut child = Command::new(&path)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("orbdiag should be spawnable");

    let stdout_pipe = child.stdout.take().expect("piped stdout");
    let collected = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let sink = std::sync::Arc::clone(&collected);
    std::thread::spawn(move || {
        use std::io::Read;
        let mut stdout_pipe = stdout_pipe;
        let mut bytes = [0u8; 4096];
        loop {
            match stdout_pipe.read(&mut bytes) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    let mut sink = sink.lock().unwrap_or_else(|e| e.into_inner());
                    sink.push_str(&String::from_utf8_lossy(&bytes[..n]));
                }
            }
        }
    });

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
    let found = loop {
        if collected
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains("verified_pairs")
        {
            break true;
        }
        if std::time::Instant::now() > deadline {
            break false;
        }
        if let Ok(Some(_)) = child.try_wait() {
            // It exited without reporting; give the reader a moment to drain
            // whatever it wrote on the way out.
            std::thread::sleep(std::time::Duration::from_millis(500));
            break collected
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .contains("verified_pairs");
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    };

    let _ = child.kill();
    let _ = child.wait();

    let stdout = collected.lock().unwrap_or_else(|e| e.into_inner()).clone();
    assert!(
        found,
        "orbdiag did not reach its first report; it may have failed early.\n{}",
        stdout.lines().take(5).collect::<Vec<_>>().join("\n")
    );
}
