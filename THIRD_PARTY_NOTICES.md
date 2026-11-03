# Third-party notices

RETINA is licensed under the [MIT license](LICENSE). Some of its content is
derived from other open-source projects, whose licences require their notices
to travel with it. Those parts stay under their original terms, listed here.

## OpenCV

- Project: [OpenCV](https://github.com/opencv/opencv), Copyright (C) 2000-2026
  the OpenCV Foundation and contributors.
- Licence: Apache License 2.0 (OpenCV 4.5 and later), full text in
  [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt).
- What RETINA uses:
  - `crates/objdetect/src/aruco_tables.rs`: the predefined ArUco and AprilTag
    dictionaries, generated from OpenCV's
    `modules/objdetect/src/aruco/predefined_dictionaries.hpp` (the AprilTag
    families also carry the AprilTag notice below).
  - Algorithms whose doc comments say they follow or port a specific OpenCV
    source file (for example `detectMarkers`, `cornerSubPix`, Farneback and
    TV-L1 optical flow, `findContours`). They are reimplemented in Rust to
    reproduce OpenCV's results; the doc comment names the OpenCV file.
- Changes: the data were re-encoded as Rust byte tables; the algorithms were
  rewritten in Rust (and WGSL for GPU kernels).

## AprilTag

- Project: [AprilTag](https://github.com/AprilRobotics/apriltag), Copyright (C)
  2013-2016 The Regents of The University of Michigan.
- Licence: BSD 2-Clause (below).
- What RETINA uses: the tag16h5, tag25h9, tag36h10 and tag36h11 code families
  in `crates/objdetect/src/aruco_tables.rs`, taken from OpenCV's copy of them.

```text
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Reference values in tests

Parity tests store numbers produced by OpenCV, Open3D, SciPy and Matplotlib.
These are measured outputs, not code from those projects; the scripts that
produced them are next to the tests.

## Dependencies

Crates RETINA depends on are under their own licences, all permissive (MIT,
Apache-2.0, BSD, ISC, Zlib, Unicode-3.0, Unlicense, and the font licences of
egui's bundled fonts). `deny.toml` lists what is allowed, and CI checks it with
`cargo deny`.
