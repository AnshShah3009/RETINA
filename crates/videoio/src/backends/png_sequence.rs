use crate::{Result, VideoCapture, VideoError, VideoWriter};
use image::GrayImage;
use std::cmp::Ordering;
use std::fs;
use std::path::{Path, PathBuf};

/// Zero-padding the writer uses for frame indices.
const FRAME_DIGITS: usize = 6;

/// Compare two strings, ordering runs of ASCII digits by numeric value instead
/// of by byte value. Frame sequences are named both `frame_000010.png` (written
/// by [`PngSequenceWriter`]) and `frame10.png` (what most tools produce, e.g.
/// `ffmpeg -i in.mp4 frame%d.png`); a byte-wise sort reads the latter as
/// 1, 10, 11, …, 2, 20, … and silently shuffles the sequence.
///
/// Ties are impossible between distinct names: equal digit runs are ordered by
/// their leading zeros, and total order is completed by the caller's
/// byte-wise fallback.
fn natural_cmp(a: &str, b: &str) -> Ordering {
    let (ab, bb) = (a.as_bytes(), b.as_bytes());
    let (mut i, mut j) = (0usize, 0usize);

    while i < ab.len() && j < bb.len() {
        if ab[i].is_ascii_digit() && bb[j].is_ascii_digit() {
            let (i0, j0) = (i, j);
            while i < ab.len() && ab[i].is_ascii_digit() {
                i += 1;
            }
            while j < bb.len() && bb[j].is_ascii_digit() {
                j += 1;
            }
            let (da, db) = (&a[i0..i], &b[j0..j]);
            // Strip leading zeros, then compare by length before contents:
            // without the leading zeros neither operand can start with a '0',
            // so the longer string is the larger number.
            let (sa, sb) = (da.trim_start_matches('0'), db.trim_start_matches('0'));
            let ord = sa.len().cmp(&sb.len()).then_with(|| sa.cmp(sb));
            if ord != Ordering::Equal {
                return ord;
            }
            // Same numeric value: fewer leading zeros first (total order).
            let ord = da.len().cmp(&db.len());
            if ord != Ordering::Equal {
                return ord;
            }
        } else {
            let ord = ab[i].cmp(&bb[j]);
            if ord != Ordering::Equal {
                return ord;
            }
            i += 1;
            j += 1;
        }
    }

    (ab.len() - i).cmp(&(bb.len() - j))
}

/// True for the raster formats this backend reads, whatever the case of the
/// extension: `frame.png` and `frame.PNG` are the same file format, and a
/// case-sensitive filter silently truncates a mixed directory.
fn is_frame_path(path: &Path) -> bool {
    match path.extension() {
        Some(ext) => {
            let ext = ext.to_string_lossy();
            ext.eq_ignore_ascii_case("png")
                || ext.eq_ignore_ascii_case("jpg")
                || ext.eq_ignore_ascii_case("jpeg")
        }
        None => false,
    }
}

/// Index one past the highest `{prefix}_{digits}.png` already in `directory`.
///
/// A writer that always restarted at zero overwrote the head of an existing
/// sequence and left the older tail in place, so the directory held two
/// interleaved sequences and a reader returned a sequence that was never
/// written. Numbering continues instead.
fn next_frame_index(directory: &Path, prefix: &str) -> Result<usize> {
    let mut next = 0usize;
    let entries = match fs::read_dir(directory) {
        Ok(entries) => entries,
        // Nothing to continue after.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(VideoError::Io(e)),
    };

    for entry in entries {
        let name = entry.map_err(VideoError::Io)?.file_name();
        let Some(rest) = name
            .to_string_lossy()
            .strip_prefix(prefix)
            .map(str::to_owned)
        else {
            continue;
        };
        let Some(rest) = rest.strip_prefix('_') else {
            continue;
        };
        if let Some(index) = rest
            .strip_suffix(".png")
            .and_then(|digits| digits.parse::<usize>().ok())
        {
            next = next.max(index + 1);
        }
    }

    Ok(next)
}

#[derive(Debug)]
pub struct PngSequenceWriter {
    directory: PathBuf,
    prefix: String,
    frame_count: usize,
}

impl PngSequenceWriter {
    /// Opens `directory` (creating it if needed) and continues the sequence
    /// already present there: the next frame is numbered after the highest
    /// `{prefix}_NNNNNN.png`, so a second writer appends rather than silently
    /// overwriting the frames an earlier writer produced.
    pub fn new(directory: &Path, prefix: &str) -> Result<Self> {
        if !directory.exists() {
            fs::create_dir_all(directory).map_err(VideoError::Io)?;
        }

        Ok(Self {
            directory: directory.to_path_buf(),
            prefix: prefix.to_string(),
            frame_count: next_frame_index(directory, prefix)?,
        })
    }
}

impl VideoWriter for PngSequenceWriter {
    fn write(&mut self, frame: &GrayImage) -> Result<()> {
        let filename = format!(
            "{}_{:0width$}.png",
            self.prefix,
            self.frame_count,
            width = FRAME_DIGITS
        );
        let path = self.directory.join(filename);

        frame
            .save(&path)
            .map_err(|e| VideoError::Backend(format!("Failed to save frame: {}", e)))?;
        self.frame_count += 1;
        Ok(())
    }
}

#[derive(Debug)]
pub struct PngSequenceCapture {
    files: Vec<PathBuf>,
    current_index: usize,
}

impl PngSequenceCapture {
    pub fn new<P: AsRef<Path>>(directory: P) -> Result<Self> {
        let mut files = Vec::new();
        if directory.as_ref().is_dir() {
            for entry in fs::read_dir(directory).map_err(VideoError::Io)? {
                let entry = entry.map_err(VideoError::Io)?;
                let path = entry.path();
                if path.is_file() && is_frame_path(&path) {
                    files.push(path);
                }
            }
        }
        // Natural order, so `frame10.png` follows `frame9.png`. Paths are
        // unique, so the fallback makes the order total.
        files.sort_by(|a, b| {
            natural_cmp(&a.to_string_lossy(), &b.to_string_lossy()).then_with(|| a.cmp(b))
        });

        if files.is_empty() {
            return Err(VideoError::Backend(
                "No image files found in directory".to_string(),
            ));
        }

        Ok(Self {
            files,
            current_index: 0,
        })
    }
}

impl VideoCapture for PngSequenceCapture {
    fn is_opened(&self) -> bool {
        !self.files.is_empty()
    }

    fn grab(&mut self) -> Result<()> {
        if self.current_index < self.files.len() {
            Ok(())
        } else {
            Err(VideoError::CaptureFailed("End of sequence".to_string()))
        }
    }

    fn retrieve(&mut self) -> Result<GrayImage> {
        if self.current_index < self.files.len() {
            let path = &self.files[self.current_index];
            let img = image::open(path)
                .map_err(|e| {
                    VideoError::Backend(format!("Failed to open image {}: {}", path.display(), e))
                })?
                .to_luma8();
            self.current_index += 1;
            Ok(img)
        } else {
            Err(VideoError::CaptureFailed("End of sequence".to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn natural_order_of_unpadded_numbers() {
        let mut names = vec!["f10.png", "f2.png", "f1.png", "f100.png", "f20.png"];
        names.sort_by(|a, b| natural_cmp(a, b));
        assert_eq!(
            names,
            vec!["f1.png", "f2.png", "f10.png", "f20.png", "f100.png"]
        );
    }

    #[test]
    fn natural_order_of_padded_numbers_matches_byte_order() {
        let mut names = vec!["f_000010.png", "f_000002.png", "f_000001.png"];
        let mut byte_sorted = names.clone();
        byte_sorted.sort();
        names.sort_by(|a, b| natural_cmp(a, b));
        assert_eq!(names, byte_sorted);
        assert_eq!(names, vec!["f_000001.png", "f_000002.png", "f_000010.png"]);
    }

    #[test]
    fn natural_order_is_total() {
        // Distinct names, including equal numeric values, must never compare
        // equal — `sort_by` needs a total order.
        let names = [
            "a1.png", "a01.png", "a001.png", "a1.jpg", "a2.png", "b1.png",
        ];
        for x in names {
            for y in names {
                let ord = natural_cmp(x, y);
                if x == y {
                    assert_eq!(ord, Ordering::Equal, "{} vs {}", x, y);
                } else {
                    assert_ne!(ord, Ordering::Equal, "{} vs {}", x, y);
                    assert_eq!(ord, natural_cmp(y, x).reverse(), "{} vs {}", x, y);
                }
            }
        }
        assert_eq!(natural_cmp("a1.png", "a01.png"), Ordering::Less);
    }

    #[test]
    fn extensions_are_matched_case_insensitively() {
        assert!(is_frame_path(Path::new("a.png")));
        assert!(is_frame_path(Path::new("a.PNG")));
        assert!(is_frame_path(Path::new("a.JpEg")));
        assert!(!is_frame_path(Path::new("a.gif")));
        assert!(!is_frame_path(Path::new("a")));
    }

    #[test]
    fn next_index_continues_the_sequence() {
        let dir = std::env::temp_dir().join(format!(
            "cv-videoio-next-index-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        assert_eq!(next_frame_index(&dir, "f").unwrap(), 0);

        for name in [
            "f_000000.png",
            "f_000001.png",
            "f_000009.png",
            "other_000099.png",
            "f_000099.jpg",
            "f_notdigits.png",
        ] {
            fs::write(dir.join(name), b"x").unwrap();
        }
        assert_eq!(next_frame_index(&dir, "f").unwrap(), 10);
        // An empty directory that does not exist yet is not an error either.
        assert_eq!(next_frame_index(&dir.join("missing"), "f").unwrap(), 0);

        let _ = fs::remove_dir_all(&dir);
    }
}
