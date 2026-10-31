use crate::context::BorderMode;
use cv_core::Float;

/// Map a single coordinate using the given border mode.
/// Returns `None` when the pixel should use the constant fill value.
pub(crate) fn map_border_coord_1d<T: Float>(
    coord: isize,
    len: usize,
    mode: &BorderMode<T>,
) -> Option<usize> {
    let n = len as isize;
    if n <= 0 {
        return None;
    }
    match mode {
        BorderMode::Constant(_) => {
            if coord < 0 || coord >= n {
                None
            } else {
                Some(coord as usize)
            }
        }
        BorderMode::Replicate => Some(coord.clamp(0, n - 1) as usize),
        BorderMode::Wrap => {
            let mut c = coord % n;
            if c < 0 {
                c += n;
            }
            Some(c as usize)
        }
        BorderMode::Reflect => {
            // OpenCV's BORDER_REFLECT: `fedcba|abcdefgh|hgfedcb` - the edge
            // sample *is* repeated, so index -1 maps to 0 and n maps to n-1.
            // Folded over a period of `2n` so coordinates more than one image
            // width outside still land in range (the same formula as
            // `cv-imgproc`'s border mapping).
            if n == 1 {
                return Some(0);
            }
            let period = 2 * n;
            let mut c = coord.rem_euclid(period);
            if c >= n {
                c = period - c - 1;
            }
            Some(c as usize)
        }
        BorderMode::Reflect101 => {
            // OpenCV's BORDER_REFLECT_101: `gfedcb|abcdefgh|gfedcba` - the edge
            // sample is *not* repeated, so index -1 maps to 1 and n maps to n-2.
            if n == 1 {
                return Some(0);
            }
            let period = 2 * n - 2;
            let mut c = coord.rem_euclid(period);
            if c >= n {
                c = period - c;
            }
            Some(c as usize)
        }
    }
}

/// Map (x, y) using border mode. Returns `Some((ix, iy))` or `None` for constant fill.
pub(crate) fn map_border_coord<T: Float>(
    x: isize,
    y: isize,
    w: usize,
    h: usize,
    mode: &BorderMode<T>,
) -> Option<(usize, usize)> {
    match (
        map_border_coord_1d(x, w, mode),
        map_border_coord_1d(y, h, mode),
    ) {
        (Some(ix), Some(iy)) => Some((ix, iy)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `BorderMode::Reflect` follows OpenCV's `BORDER_REFLECT`.
    ///
    /// `fedcba|abcdefgh|hgfedcb` - the edge sample is repeated, so -1 maps to
    /// 0 and n maps to n-1. Expected values are `cv2.copyMakeBorder` on
    /// `[0, 1, 2, 3, 4]` with `BORDER_REFLECT`.
    ///
    /// cb25ce4 had this table and the Reflect101 one swapped: its comments
    /// described OpenCV's two modes back to front, so the tests pinned the
    /// swap rather than catching it.
    #[test]
    fn reflect_matches_opencv_repeating_the_edge() {
        let expect: &[(isize, usize)] = &[
            (-7, 3),
            (-6, 4),
            (-5, 4),
            (-3, 2),
            (-2, 1),
            (-1, 0),
            (0, 0),
            (1, 1),
            (4, 4),
            (5, 4),
            (6, 3),
            (7, 2),
            (9, 0),
            (14, 4),
        ];
        for (coord, want) in expect {
            let got = map_border_coord_1d::<f32>(*coord, 5, &BorderMode::Reflect);
            assert_eq!(got, Some(*want), "Reflect({coord}) should be {want}");
        }
    }

    /// `BorderMode::Reflect101` follows OpenCV's `BORDER_REFLECT_101`.
    ///
    /// `gfedcb|abcdefgh|gfedcba` - the edge sample is not repeated, so -1 maps
    /// to 1 and n maps to n-2. Expected values are `cv2.copyMakeBorder` on
    /// `[0, 1, 2, 3, 4]` with `BORDER_REFLECT_101`.
    #[test]
    fn reflect101_matches_opencv_without_repeating_the_edge() {
        let expect: &[(isize, usize)] = &[
            (-7, 1),
            (-5, 3),
            (-4, 4),
            (-3, 3),
            (-2, 2),
            (-1, 1),
            (0, 0),
            (1, 1),
            (4, 4),
            (5, 3),
            (6, 2),
            (7, 1),
            (8, 0),
        ];
        for (coord, want) in expect {
            let got = map_border_coord_1d::<f32>(*coord, 5, &BorderMode::Reflect101);
            assert_eq!(got, Some(*want), "Reflect101({coord}) should be {want}");
        }
    }

    /// The two reflect modes must not be the same function.
    ///
    /// A regression guard rather than a property test: they were identical for
    /// their entire life, and any future folding that collapses them again
    /// would pass both tests above if those only checked in-range indices.
    #[test]
    fn the_two_reflect_modes_differ_where_they_should() {
        let n = 5;
        let differ = (-6..12)
            .filter(|c| {
                map_border_coord_1d::<f32>(*c, n, &BorderMode::Reflect)
                    != map_border_coord_1d::<f32>(*c, n, &BorderMode::Reflect101)
            })
            .count();
        assert!(
            differ > 0,
            "Reflect and Reflect101 produce identical results everywhere, so \\
             one of the two enum variants is meaningless"
        );
    }

    /// Indices inside the image are the identity for every border mode.
    #[test]
    fn in_range_indices_are_the_identity() {
        for n in [2usize, 3, 5, 17] {
            for c in 0..n {
                for mode in [
                    BorderMode::Reflect,
                    BorderMode::Reflect101,
                    BorderMode::Replicate,
                    BorderMode::Wrap,
                ] {
                    assert_eq!(
                        map_border_coord_1d::<f32>(c as isize, n, &mode),
                        Some(c),
                        "{mode:?} moved in-range index {c} of {n}"
                    );
                }
            }
        }
    }
}
