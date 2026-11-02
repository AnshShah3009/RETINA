"""Regenerate the OpenCV reference fixtures for tests/markers_opencv_parity.rs.

Run from this directory with OpenCV 4.13 (`python3 generate.py`). Writes the
PNG images next to this script and prints the expected values that the test
embeds (dictionary checksums, detected markers and ChArUco corners).
"""

import cv2
import numpy as np

A = cv2.aruco

DICTS = [
    ("Dict4x4_50", A.DICT_4X4_50),
    ("Dict4x4_100", A.DICT_4X4_100),
    ("Dict4x4_250", A.DICT_4X4_250),
    ("Dict5x5_50", A.DICT_5X5_50),
    ("Dict5x5_100", A.DICT_5X5_100),
    ("Dict5x5_250", A.DICT_5X5_250),
    ("Dict6x6_50", A.DICT_6X6_50),
    ("Dict6x6_100", A.DICT_6X6_100),
    ("Dict6x6_250", A.DICT_6X6_250),
    ("AprilTag16h5", A.DICT_APRILTAG_16h5),
    ("AprilTag25h9", A.DICT_APRILTAG_25h9),
    ("AprilTag36h10", A.DICT_APRILTAG_36h10),
    ("AprilTag36h11", A.DICT_APRILTAG_36h11),
]


def fnv1a(data, h=0xCBF29CE484222325):
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def dictionary_checksums():
    """FNV-1a over every marker's payload grid (row-major, 1 = white)."""
    print("// dictionary checksums")
    for name, const in DICTS:
        d = A.getPredefinedDictionary(const)
        h = 0xCBF29CE484222325
        for i in range(d.bytesList.shape[0]):
            bits = d.getBitsFromByteList(d.bytesList[i : i + 1], d.markerSize)
            h = fnv1a(bits.ravel().tolist(), h)
        print(f"(ArucoDictionary::{name}, 0x{h:016x}),")


def paste(canvas, img, x, y):
    canvas[y : y + img.shape[0], x : x + img.shape[1]] = img


def scene(markers, size, warp):
    """Markers `(dict, id, side, x, y, quarter_turns_ccw)` on a white canvas,
    then a perspective warp of the whole canvas (corners moved by `warp`)."""
    w, h = size
    canvas = np.full((h, w), 255, np.uint8)
    for const, mid, side, x, y, k in markers:
        d = A.getPredefinedDictionary(const)
        paste(canvas, np.rot90(d.generateImageMarker(mid, side), k), x, y)
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = src + np.float32(warp)
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        canvas, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=255
    )


def print_detections(name, img, dicts):
    for dname, const in dicts:
        det = A.ArucoDetector(A.getPredefinedDictionary(const), A.DetectorParameters())
        corners, ids, _ = det.detectMarkers(img)
        print(f"// {name} {dname}")
        rows = []
        if ids is not None:
            for c, i in zip(corners, ids.ravel()):
                pts = ", ".join(f"({x:.4f}, {y:.4f})" for x, y in c.reshape(4, 2))
                rows.append((int(i), pts))
        for i, pts in sorted(rows):
            print(f"({i}, [{pts}]),")


def main():
    dictionary_checksums()

    aruco = scene(
        [
            (A.DICT_4X4_50, 3, 60, 30, 30, 0),
            (A.DICT_4X4_50, 17, 72, 150, 40, 1),
            (A.DICT_4X4_50, 42, 54, 290, 50, 2),
            (A.DICT_6X6_250, 0, 80, 40, 160, 3),
            (A.DICT_6X6_250, 123, 64, 170, 170, 0),
            (A.DICT_6X6_250, 249, 72, 290, 160, 1),
        ],
        (400, 280),
        [[6, 10], [-14, 4], [-8, -12], [12, -6]],
    )
    cv2.imwrite("aruco_scene.png", aruco)
    print_detections(
        "aruco_scene.png",
        aruco,
        [("Dict4x4_50", A.DICT_4X4_50), ("Dict6x6_250", A.DICT_6X6_250)],
    )

    april = scene(
        [
            (A.DICT_APRILTAG_16h5, 4, 60, 30, 30, 1),
            (A.DICT_APRILTAG_25h9, 20, 63, 150, 30, 0),
            (A.DICT_APRILTAG_36h10, 1500, 64, 30, 150, 2),
            (A.DICT_APRILTAG_36h11, 300, 72, 150, 150, 3),
            (A.DICT_APRILTAG_36h11, 586, 56, 270, 90, 0),
        ],
        (360, 260),
        [[-6, 8], [10, -4], [6, 9], [-9, -5]],
    )
    cv2.imwrite("apriltag_scene.png", april)
    print_detections(
        "apriltag_scene.png",
        april,
        [
            ("AprilTag16h5", A.DICT_APRILTAG_16h5),
            ("AprilTag25h9", A.DICT_APRILTAG_25h9),
            ("AprilTag36h10", A.DICT_APRILTAG_36h10),
            ("AprilTag36h11", A.DICT_APRILTAG_36h11),
        ],
    )

    # `draw_charuco_board(board, 40)`.
    b = A.CharucoBoard((5, 7), 0.04, 0.03, A.getPredefinedDictionary(A.DICT_4X4_50))
    board = b.generateImage((200, 280))
    cv2.imwrite("charuco_5x7_4x4_50.png", board)
    # `generate_charuco_board_image(board, 500, 380, 17)`: the board is fitted
    # into the area inside the margin and centred.
    b = A.CharucoBoard((6, 4), 1.0, 0.7, A.getPredefinedDictionary(A.DICT_5X5_100))
    cv2.imwrite("charuco_6x4_5x5_100_margin.png", b.generateImage((500, 380), None, 17))
    b = A.CharucoBoard((7, 5), 0.035, 0.021, A.getPredefinedDictionary(A.DICT_6X6_250))
    cv2.imwrite("charuco_7x5_6x6_250_margin.png", b.generateImage((613, 452), None, 9))

    # ChArUco corners of the first board inside a white frame, from
    # OpenCV's CharucoDetector (default parameters).
    b = A.CharucoBoard((5, 7), 0.04, 0.03, A.getPredefinedDictionary(A.DICT_4X4_50))
    framed = cv2.copyMakeBorder(board, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    corners, ids, _, _ = A.CharucoDetector(b).detectBoard(framed)
    print("// charuco corners, board in a 20 px frame")
    for c, i in sorted(zip(corners.reshape(-1, 2).tolist(), ids.ravel().tolist()), key=lambda t: t[1]):
        print(f"({i}, ({c[0]:.4f}, {c[1]:.4f})),")


if __name__ == "__main__":
    main()
