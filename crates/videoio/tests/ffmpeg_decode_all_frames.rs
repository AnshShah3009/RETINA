//! The FFmpeg backend must return every frame of a stream whose decoder has
//! reordering delay (B-frames). It used to send one packet and receive at most
//! one frame per packet; with output still pending, the next `send_packet`
//! failed with EAGAIN and the stream ended early as `CaptureFailed`
//! (GAPS g-i11). Fixtures are generated with the system `ffmpeg` CLI; the test
//! skips when it is absent. Runs only with `--features ffmpeg`.
#![cfg(feature = "ffmpeg")]

use cv_videoio::VideoCapture;
use cv_videoio::backends::NativeFfmpegCapture;
use std::process::Command;

fn make(name: &str, args: &[&str]) -> Option<std::path::PathBuf> {
    let path = std::env::temp_dir().join(format!("cv_videoio_{}_{name}", std::process::id()));
    let ok = Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=96x64:rate=25:duration=1.2",
        ])
        .args(args)
        .arg(&path)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    ok.then_some(path)
}

fn count_frames(path: &std::path::Path) -> usize {
    let mut cap = NativeFfmpegCapture::new(path).expect("open");
    let mut n = 0;
    let mut last: Option<image::GrayImage> = None;
    while cap.grab().is_ok() {
        let frame = cap.retrieve().expect("retrieve after grab");
        assert_eq!(frame.dimensions(), (96, 64));
        if let Some(prev) = &last {
            assert_ne!(
                prev.as_raw(),
                frame.as_raw(),
                "frame {n} repeats the previous one"
            );
        }
        last = Some(frame);
        n += 1;
    }
    n
}

#[test]
fn every_frame_of_a_b_frame_stream_is_returned() {
    // 1.2 s at 25 fps = 30 frames; testsrc changes every frame.
    let cases: [(&str, &[&str]); 2] = [
        (
            "h264_bframes.mp4",
            &["-c:v", "libx264", "-bf", "3", "-g", "12"],
        ),
        (
            "mpeg4_bframes.avi",
            &["-c:v", "mpeg4", "-bf", "2", "-g", "12"],
        ),
    ];
    for (name, args) in cases {
        let Some(path) = make(name, args) else {
            eprintln!("skipping {name}: ffmpeg CLI or encoder unavailable");
            continue;
        };
        assert_eq!(count_frames(&path), 30, "{name}");
        let _ = std::fs::remove_file(path);
    }
}
