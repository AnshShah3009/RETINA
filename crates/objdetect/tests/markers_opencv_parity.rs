//! `cv_objdetect::markers` against OpenCV 4.13's `cv2.aruco`.
//!
//! The PNG fixtures and the expected values below come from
//! `tests/fixtures/markers/generate.py`: dictionaries, markers and ChArUco
//! boards drawn by OpenCV, and what OpenCV's `ArucoDetector` and
//! `CharucoDetector` find in them.

use cv_core::{CpuTensor, TensorShape};
use cv_objdetect::aruco::ArucoDetector;
use cv_objdetect::charuco::{CharucoBoard as TensorBoard, CharucoDetector, CharucoParameters};
use cv_objdetect::markers::*;
use image::{GrayImage, Luma};

fn fixture(name: &str) -> GrayImage {
    let path = format!(
        "{}/tests/fixtures/markers/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    image::open(&path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .into_luma8()
}

type Marker = (u16, [(f32, f32); 4]);

/// Ids and corners (marker's top-left first), sorted by id.
fn found(dets: impl IntoIterator<Item = (u16, [nalgebra::Point2<f32>; 4])>) -> Vec<Marker> {
    let mut out: Vec<Marker> = dets
        .into_iter()
        .map(|(id, c)| (id, c.map(|p| (p.x, p.y))))
        .collect();
    out.sort_by_key(|m| m.0);
    out
}

/// FNV-1a over every marker's payload grid, row-major, 1 = white: OpenCV's
/// `Dictionary::getBitsFromByteList` for every id of every dictionary.
#[test]
fn every_dictionary_marker_matches_opencv() {
    let expected = [
        (ArucoDictionary::Dict4x4_50, 0xe2417ad5bd52fcf1u64),
        (ArucoDictionary::Dict4x4_100, 0xa2fe095b5fd1e534),
        (ArucoDictionary::Dict4x4_250, 0x2991763e01083345),
        (ArucoDictionary::Dict5x5_50, 0xd59d38d6a61471ae),
        (ArucoDictionary::Dict5x5_100, 0x04110637e3c53b98),
        (ArucoDictionary::Dict5x5_250, 0x8c9d9e1feb6e8172),
        (ArucoDictionary::Dict6x6_50, 0x69856416627da313),
        (ArucoDictionary::Dict6x6_100, 0x23a1c555c1499295),
        (ArucoDictionary::Dict6x6_250, 0xf65af45d0f582e69),
        (ArucoDictionary::AprilTag16h5, 0xcae940bac39c3325),
        (ArucoDictionary::AprilTag25h9, 0xce20954833c95bc5),
        (ArucoDictionary::AprilTag36h10, 0x1e6c64f9640270c7),
        (ArucoDictionary::AprilTag36h11, 0x41439cfa91d9b1be),
    ];
    for (dict, want) in expected {
        let mut h = 0xcbf29ce484222325u64;
        for id in 0..dict.dict_size() {
            for bit in dict.get_marker(id).unwrap().into_iter().flatten() {
                h = (h ^ u64::from(bit)).wrapping_mul(0x100000001b3);
            }
        }
        assert_eq!(h, want, "{dict:?}");
    }
}

/// Markers drawn by `generateImageMarker`, turned and seen in perspective:
/// the same ids and corners as OpenCV's `detectMarkers`.
#[test]
fn reads_opencv_aruco_markers() {
    let img = fixture("aruco_scene.png");
    let cases: [(ArucoDictionary, Vec<Marker>); 2] = [
        (
            ArucoDictionary::Dict4x4_50,
            vec![
                (3, [(35.0, 38.0), (91.0, 37.0), (93.0, 92.0), (37.0, 94.0)]),
                (
                    17,
                    [(151.0, 113.0), (149.0, 45.0), (217.0, 44.0), (218.0, 111.0)],
                ),
                (
                    42,
                    [(334.0, 102.0), (284.0, 103.0), (282.0, 53.0), (333.0, 52.0)],
                ),
            ],
        ),
        (
            ArucoDictionary::Dict6x6_250,
            vec![
                (
                    0,
                    [(123.0, 159.0), (124.0, 234.0), (49.0, 235.0), (47.0, 160.0)],
                ),
                (
                    123,
                    [
                        (171.0, 168.0),
                        (231.0, 167.0),
                        (233.0, 226.0),
                        (172.0, 227.0),
                    ],
                ),
                (
                    249,
                    [
                        (287.0, 224.0),
                        (285.0, 156.0),
                        (352.0, 155.0),
                        (354.0, 222.0),
                    ],
                ),
            ],
        ),
    ];
    for (dict, want) in cases {
        let dets = detect_aruco_markers(&img, dict).unwrap();
        assert_eq!(
            found(dets.iter().map(|d| (d.id, d.corners))),
            want,
            "{dict:?}"
        );
    }
    // Rotations: the scene turned the markers 0, 1 and 2 quarter turns
    // counter-clockwise (4x4) and 3, 0, 1 (6x6).
    let rot = |dict| {
        let mut r: Vec<(u16, u8)> = detect_aruco_markers(&img, dict)
            .unwrap()
            .iter()
            .map(|d| (d.id, d.rotation_cw))
            .collect();
        r.sort();
        r
    };
    assert_eq!(rot(ArucoDictionary::Dict4x4_50), [(3, 0), (17, 1), (42, 2)]);
    assert_eq!(
        rot(ArucoDictionary::Dict6x6_250),
        [(0, 3), (123, 0), (249, 1)]
    );
}

/// AprilTags drawn by OpenCV (`DICT_APRILTAG_*`): the same ids and corners
/// as OpenCV's `detectMarkers` with each family's dictionary.
#[test]
fn reads_opencv_apriltags() {
    let img = fixture("apriltag_scene.png");
    let cases: [(AprilTagFamily, Vec<Marker>); 4] = [
        (
            AprilTagFamily::Tag16h5,
            vec![(4, [(21.0, 92.0), (22.0, 36.0), (80.0, 34.0), (79.0, 92.0)])],
        ),
        (
            AprilTagFamily::Tag25h9,
            vec![(
                20,
                [(141.0, 33.0), (206.0, 31.0), (205.0, 94.0), (140.0, 94.0)],
            )],
        ),
        (
            AprilTagFamily::Tag36h10,
            vec![(
                1500,
                [(81.0, 213.0), (20.0, 211.0), (21.0, 151.0), (82.0, 151.0)],
            )],
        ),
        (
            AprilTagFamily::Tag36h11,
            vec![
                (
                    300,
                    [
                        (214.0, 152.0),
                        (213.0, 224.0),
                        (138.0, 222.0),
                        (139.0, 152.0),
                    ],
                ),
                (
                    586,
                    [(267.0, 91.0), (329.0, 91.0), (328.0, 148.0), (267.0, 148.0)],
                ),
            ],
        ),
    ];
    for (family, want) in cases {
        let dets = detect_apriltags(&img, family).unwrap();
        assert!(dets.iter().all(|d| d.hamming == 0));
        assert_eq!(
            found(dets.iter().map(|d| (d.id, d.corners))),
            want,
            "{family:?}"
        );
    }
}

/// Boards drawn here are OpenCV's `CharucoBoard::generateImage` images,
/// pixel for pixel: square colours, marker ids and placement, the marker
/// inset and the fitting of the board inside a margin.
#[test]
fn draws_opencv_charuco_boards() {
    let board = create_charuco_board(5, 7, 0.04, 0.03, ArucoDictionary::Dict4x4_50).unwrap();
    assert_eq!(
        draw_charuco_board(&board, 40).unwrap(),
        fixture("charuco_5x7_4x4_50.png")
    );

    let board = create_charuco_board(6, 4, 1.0, 0.7, ArucoDictionary::Dict5x5_100).unwrap();
    assert_eq!(
        generate_charuco_board_image(&board, 500, 380, 17).unwrap(),
        fixture("charuco_6x4_5x5_100_margin.png")
    );

    let board = create_charuco_board(7, 5, 0.035, 0.021, ArucoDictionary::Dict6x6_250).unwrap();
    assert_eq!(
        generate_charuco_board_image(&board, 613, 452, 9).unwrap(),
        fixture("charuco_7x5_6x6_250_margin.png")
    );
}

/// A single marker image is the matching `generateImageMarker` output: the
/// 4x4 board's marker 0 sits at (45, 5) in the 5x7 board image above
/// (square 40 px, marker 30 px, inset 5 px).
#[test]
fn draws_opencv_markers() {
    let board = fixture("charuco_5x7_4x4_50.png");
    // `draw_aruco_marker` sizes are multiples of the cell count; 30 px is
    // 6 cells of 5 px.
    let marker = draw_aruco_marker(ArucoDictionary::Dict4x4_50, 0, 5).unwrap();
    for y in 0..30 {
        for x in 0..30 {
            assert_eq!(
                marker.get_pixel(x, y),
                board.get_pixel(45 + x, 5 + y),
                "pixel ({x}, {y})"
            );
        }
    }

    // Sizes that are not a multiple of the cell count follow OpenCV's
    // `INTER_NEAREST` resize: at 34 px, pixel 17 still shows cell 2
    // (`floor(17 * (1 / (34 / 6)))`), not cell 3 (`17 * 6 / 34`). Cells 2
    // and 3 of marker 0 differ in its first payload row.
    let m = cv_objdetect::aruco::draw_marker(ArucoDictionary::Dict4x4_50, 0, 34).unwrap();
    let px = m.as_slice().unwrap();
    for i in 0..34 {
        assert_eq!(px[i * 34 + 17], px[i * 34 + 16], "row {i}");
        assert_eq!(px[17 * 34 + i], px[16 * 34 + i], "column {i}");
    }
    assert_ne!(px[7 * 34 + 16], px[7 * 34 + 18]);
}

/// `markers` boards are boards of `charuco::CharucoBoard`: on the board
/// drawn here (identical to OpenCV's), `charuco::CharucoDetector` finds every
/// corner OpenCV's `CharucoDetector` finds, with the same ids, on the true
/// corner positions.
#[test]
fn charuco_detector_reads_drawn_boards() {
    let board = create_charuco_board(5, 7, 0.04, 0.03, ArucoDictionary::Dict4x4_50).unwrap();
    let img = draw_charuco_board(&board, 40).unwrap();
    let mut framed = GrayImage::from_pixel(240, 320, Luma([255]));
    image::imageops::replace(&mut framed, &img, 20, 20);

    // OpenCV's corners on the same image (`detectBoard`, default parameters).
    let opencv: [(i32, (f32, f32)); 24] = [
        (0, (60.0182, 60.0182)),
        (1, (99.9994, 59.9994)),
        (2, (140.0182, 60.0182)),
        (3, (179.9994, 59.9994)),
        (4, (59.9994, 99.9994)),
        (5, (100.0182, 100.0182)),
        (6, (139.9994, 99.9994)),
        (7, (180.0182, 100.0182)),
        (8, (60.0182, 140.0182)),
        (9, (99.9994, 139.9994)),
        (10, (140.0182, 140.0182)),
        (11, (179.9994, 139.9994)),
        (12, (59.9994, 179.9994)),
        (13, (100.0182, 180.0182)),
        (14, (139.9994, 179.9994)),
        (15, (180.0182, 180.0182)),
        (16, (60.0182, 220.0182)),
        (17, (99.9994, 219.9994)),
        (18, (140.0182, 220.0182)),
        (19, (179.9994, 219.9994)),
        (20, (59.9994, 259.9994)),
        (21, (100.0182, 260.0182)),
        (22, (139.9994, 259.9994)),
        (23, (180.0182, 260.0182)),
    ];

    let tensor_board = TensorBoard::new(
        board.squares_x as usize,
        board.squares_y as usize,
        f64::from(board.square_length),
        f64::from(board.marker_length),
        board.dictionary,
    );
    let detector = CharucoDetector::new(
        tensor_board,
        CharucoParameters::default(),
        ArucoDetector::new(board.dictionary),
    );
    let pixels: Vec<f32> = framed.pixels().map(|p| f32::from(p[0]) / 255.0).collect();
    let tensor = CpuTensor::from_vec(pixels, TensorShape::new(1, 320, 240)).unwrap();
    let result = detector.detect(&tensor).unwrap();
    assert_eq!(result.ids, (0..24).collect::<Vec<i32>>());
    for ((id, c), (want_id, (wx, wy))) in result.ids.iter().zip(&result.corners).zip(opencv) {
        assert_eq!(*id, want_id);
        // Corner (gx, gy) is the boundary between pixels 20 + 40 g - 1 and
        // 20 + 40 g, at 19.5 + 40 g in pixel-centre coordinates.
        let (gx, gy) = ((id % 4 + 1) as f32, (id / 4 + 1) as f32);
        let (tx, ty) = (19.5 + 40.0 * gx, 19.5 + 40.0 * gy);
        assert!(
            (c[0] - tx).abs() < 0.05 && (c[1] - ty).abs() < 0.05,
            "corner {id}: {c:?} vs ({tx}, {ty})"
        );
        // OpenCV's `detectBoard` lands up to half a pixel further out (its
        // `cv2.cornerSubPix` alone converges to the true corner here).
        assert!(
            (c[0] - wx).abs() < 0.55 && (c[1] - wy).abs() < 0.55,
            "corner {id}: {c:?} vs OpenCV ({wx}, {wy})"
        );
    }
}
