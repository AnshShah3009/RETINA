use crate::context::BorderMode;
use cv_core::Float;

/// Map a single coordinate using the given border mode.
/// Returns `None` when the pixel should use the constant fill value.
pub(crate) fn map_border_coord_1d<T: Float>(coord: isize, len: usize, mode: &BorderMode<T>) -> Option<usize> {
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
            if n == 1 {
                return Some(0);
            }
            let period = 2 * n;
            let mut c = coord % period;
            if c < 0 {
                c += period;
            }
            if c >= n {
                c = period - c - 1;
            }
            Some(c as usize)
        }
        BorderMode::Reflect101 => {
            if n == 1 {
                return Some(0);
            }
            let period = 2 * n - 2;
            let mut c = coord % period;
            if c < 0 {
                c += period;
            }
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
