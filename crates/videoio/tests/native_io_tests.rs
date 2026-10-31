#![forbid(unsafe_code)]
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

/// Solid gray frame used to identify a frame by its value.
fn solid(value: u8) -> GrayImage {
    let mut img = GrayImage::new(4, 3);
    for p in img.pixels_mut() {
        *p = Luma([value]);
    }
    img
}

/// The GIF backend had no test at all. It decodes every frame up front, and
/// dropping the first frame, duplicating one, returning them out of order or
/// reporting end of stream early are all invisible to the PNG tests above.
/// The file is built here with the same `image` crate the backend decodes with,
/// so the expected frame values and count are known exactly.
#[test]
fn test_gif_capture_reads_every_frame_in_order() {
    use cv_videoio::backends::GifCapture;
    use image::codecs::gif::{GifEncoder, Repeat};
    use image::{Delay, Frame, RgbaImage};
    use std::fs::File;

    let dir = tempdir().expect("Failed to create temp dir");
    let path = dir.path().join("frames.gif");
    let values: Vec<u8> = (0..6).map(|i| i * 40).collect();

    {
        let file = File::create(&path).expect("Failed to create the gif");
        let mut encoder = GifEncoder::new(file);
        encoder.set_repeat(Repeat::Infinite).unwrap();
        for (i, value) in values.iter().enumerate() {
            let img = RgbaImage::from_pixel(8, 6, image::Rgba([*value, *value, *value, 255]));
            encoder
                .encode_frame(Frame::from_parts(
                    img,
                    0,
                    0,
                    Delay::from_numer_denom_ms(100 + i as u32, 1),
                ))
                .unwrap();
        }
    }

    let mut capture = GifCapture::new(&path).unwrap();
    assert!(capture.is_opened());

    let mut got = Vec::new();
    while let Ok(frame) = capture.read() {
        assert_eq!(frame.dimensions(), (8, 6));
        got.push(frame.get_pixel(0, 0)[0]);
    }
    assert_eq!(got, values, "each frame once, in order");

    // ...and the control: a single-frame GIF still round-trips.
    let one = dir.path().join("one.gif");
    {
        let file = File::create(&one).unwrap();
        let mut encoder = GifEncoder::new(file);
        encoder
            .encode_frame(Frame::from_parts(
                RgbaImage::from_pixel(4, 4, image::Rgba([90, 90, 90, 255])),
                0,
                0,
                Delay::from_numer_denom_ms(50, 1),
            ))
            .unwrap();
    }
    let mut capture = GifCapture::new(&one).unwrap();
    assert_eq!(capture.read().unwrap().get_pixel(0, 0)[0], 90);
    assert!(capture.read().is_err());
}

/// Reads a whole sequence, returning the value of pixel (0, 0) of each frame.
fn drain(capture: &mut PngSequenceCapture) -> Vec<u8> {
    let mut values = Vec::new();
    while let Ok(frame) = capture.read() {
        values.push(frame.get_pixel(0, 0)[0]);
    }
    values
}

/// Frame numbers of unequal width must be ordered numerically: a directory of
/// `frame1.png … frame10.png` is a sequence, and its temporal order is numeric,
/// not lexicographic. `frame10.png` sorts between `frame1.png` and `frame2.png`
/// as bytes, which silently shuffles the sequence (the reader cannot tell).
#[test]
fn test_png_sequence_orders_unpadded_frame_numbers_numerically() {
    let dir = tempdir().expect("Failed to create temp dir");
    for i in 1..=10u32 {
        // 10, 20, …, 100 — the value is the frame index, so the expected read
        // order is visible in the data itself.
        solid((i * 10) as u8)
            .save(dir.path().join(format!("frame{}.png", i)))
            .unwrap();
    }

    let mut capture = PngSequenceCapture::new(dir.path()).unwrap();
    let got = drain(&mut capture);
    let expected: Vec<u8> = (1..=10u32).map(|i| (i * 10) as u8).collect();
    assert_eq!(
        got, expected,
        "frame numbers must be read in numeric order, not byte order"
    );
}

/// Control for the test above: zero-padded names (what this crate's own writer
/// produces) must keep their exact order — the fix must not disturb them.
#[test]
fn test_png_sequence_keeps_zero_padded_order() {
    let dir = tempdir().expect("Failed to create temp dir");
    let values = [70u8, 10, 90, 30, 50];
    for (i, v) in values.iter().enumerate() {
        solid(*v)
            .save(dir.path().join(format!("frame_{:06}.png", i)))
            .unwrap();
    }

    let mut capture = PngSequenceCapture::new(dir.path()).unwrap();
    assert_eq!(drain(&mut capture), values.to_vec());
}

/// Extensions are compared case-sensitively today, so a directory holding
/// `a01.png`, `a02.PNG`, `a03.jpg` silently yields one frame instead of three:
/// the sequence is truncated and the reader reports a plain end of stream.
/// Upper- and lower-case extensions are the same image format.
#[test]
fn test_png_sequence_reads_uppercase_extensions() {
    let dir = tempdir().expect("Failed to create temp dir");
    let names = ["a01.png", "a02.PNG", "a03.Jpg"];
    for (i, name) in names.iter().enumerate() {
        solid((i as u8 + 1) * 10)
            .save(dir.path().join(name))
            .unwrap();
    }

    let mut capture = PngSequenceCapture::new(dir.path()).unwrap();
    assert_eq!(
        drain(&mut capture),
        vec![10, 20, 30],
        "a mixed-case extension must not silently drop frames from the sequence"
    );
}

/// A second writer over the same directory restarts numbering at zero and
/// silently overwrites the frames already there, so the directory ends up
/// holding two interleaved sequences: the reader returns 3 frames where 5 were
/// written, and the first two are the wrong ones.
#[test]
fn test_png_sequence_writer_does_not_clobber_existing_frames() {
    let dir = tempdir().expect("Failed to create temp dir");

    let mut writer = PngSequenceWriter::new(dir.path(), "seq").unwrap();
    for v in [10u8, 20, 30] {
        writer.write(&solid(v)).unwrap();
    }
    drop(writer);

    let mut writer = PngSequenceWriter::new(dir.path(), "seq").unwrap();
    for v in [100u8, 110] {
        writer.write(&solid(v)).unwrap();
    }
    drop(writer);

    let mut capture = PngSequenceCapture::new(dir.path()).unwrap();
    assert_eq!(
        drain(&mut capture),
        vec![10, 20, 30, 100, 110],
        "a new writer must append to the sequence, not overwrite its head"
    );
}

/// Drives the real V4L2 backend when a camera is present. CI has none, and a
/// machine where the device is busy is not a failure of this crate, so every
/// environmental failure prints a note and returns instead of failing.
///
/// The property under test is the one `start_stream` promises: the caller
/// either gets frames at the resolution it asked for, or an error — never
/// silently frames of another size (the kernel substitutes the nearest
/// supported mode for a size the device does not advertise).
#[cfg(feature = "v4l2")]
#[test]
fn test_v4l2_capture_honours_the_requested_resolution() {
    use cv_videoio::backends::V4L2Capture;

    let device = std::env::var("CV_VIDEOIO_V4L2_DEVICE").unwrap_or_else(|_| "/dev/video0".into());

    {
        let mut capture = match V4L2Capture::new(&device) {
            Ok(capture) => capture,
            Err(e) => {
                println!("skipping: no usable camera at {} ({})", device, e);
                return;
            }
        };
        if let Err(e) = capture.start_stream(640, 480) {
            println!("skipping: {} does not stream 640x480 YUYV ({})", device, e);
            return;
        }

        for _ in 0..3 {
            match capture.read() {
                Ok(frame) => assert_eq!(
                    frame.dimensions(),
                    (640, 480),
                    "a granted request must deliver frames at the requested size"
                ),
                Err(e) => {
                    println!("skipping: no frame from {} ({})", device, e);
                    return;
                }
            }
        }
    } // dropping the capture releases the device

    let mut capture = match V4L2Capture::new(&device) {
        Ok(capture) => capture,
        Err(e) => {
            println!("skipping: {} busy after first capture ({})", device, e);
            return;
        }
    };
    // 1234x567 is not a mode any UVC camera advertises: measured on an ASUS FHD
    // webcam, the kernel answers this request with 1280x720.
    match capture.start_stream(1234, 567) {
        Ok(()) => {
            println!("note: {} supports 1234x567 exactly", device);
            let frame = capture.read().expect("a stream that started must deliver");
            assert_eq!(frame.dimensions(), (1234, 567));
        }
        Err(e) => println!("note: substituted size rejected, as required: {}", e),
    }
}
