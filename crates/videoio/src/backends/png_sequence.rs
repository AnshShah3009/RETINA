use crate::{Result, VideoCapture, VideoError, VideoWriter};
use image::GrayImage;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub struct PngSequenceWriter {
    directory: PathBuf,
    prefix: String,
    frame_count: usize,
}

impl PngSequenceWriter {
    pub fn new(directory: &Path, prefix: &str) -> Result<Self> {
        if !directory.exists() {
            fs::create_dir_all(directory).map_err(VideoError::Io)?;
        }

        Ok(Self {
            directory: directory.to_path_buf(),
            prefix: prefix.to_string(),
            frame_count: 0,
        })
    }
}

impl VideoWriter for PngSequenceWriter {
    fn write(&mut self, frame: &GrayImage) -> Result<()> {
        let filename = format!("{}_{:06}.png", self.prefix, self.frame_count);
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
                if path.is_file() {
                    if let Some(ext) = path.extension() {
                        if ext == "png" || ext == "jpg" || ext == "jpeg" {
                            files.push(path);
                        }
                    }
                }
            }
        }
        files.sort();

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
