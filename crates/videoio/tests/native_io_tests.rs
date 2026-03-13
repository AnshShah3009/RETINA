use cv_videoio::{
    backends::{PngSequenceCapture, PngSequenceWriter},
    VideoCapture, VideoWriter,
};
use image::{GrayImage, Luma};
use tempfile::tempdir;

#[test]
fn test_png_sequence_roundtrip() {
    let dir = tempdir().expect("Failed to create temp dir");
    let prefix = "frame";

    // 1. Write frames
    let mut writer = PngSequenceWriter::new(dir.path(), prefix).unwrap();
    let width = 64;
    let height = 48;

    for i in 0..5 {
        let mut img = GrayImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                img.put_pixel(x, y, Luma([i as u8 * 10]));
            }
        }
        writer.write(&img).unwrap();
    }

    // 2. Read frames back
    let mut capture = PngSequenceCapture::new(dir.path()).unwrap();
    assert!(capture.is_opened());

    for i in 0..5 {
        let img = capture.read().unwrap();
        assert_eq!(img.width(), width);
        assert_eq!(img.height(), height);
        assert_eq!(img.get_pixel(0, 0)[0], i as u8 * 10);
    }

    // 3. Verify end of stream
    assert!(capture.read().is_err());
}

#[test]
fn test_png_sequence_invalid_dir() {
    let res = PngSequenceCapture::new("/non/existent/path");
    assert!(res.is_err());
}
