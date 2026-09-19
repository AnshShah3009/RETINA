//! Native FFmpeg video capture backend using ffmpeg-next

use crate::{Result, VideoCapture, VideoError};
use ffmpeg_next as ffmpeg;
use image::GrayImage;
use std::path::Path;

pub struct NativeFfmpegCapture {
    ictx: ffmpeg::format::context::Input,
    decoder: ffmpeg::decoder::Video,
    stream_index: usize,
    width: u32,
    height: u32,
    scaler: ffmpeg::software::scaling::Context,
    current_frame: Option<GrayImage>,
}

impl std::fmt::Debug for NativeFfmpegCapture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeFfmpegCapture")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish()
    }
}

impl NativeFfmpegCapture {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        ffmpeg::init().map_err(|e| VideoError::Backend(format!("FFmpeg init failed: {}", e)))?;

        let ictx = ffmpeg::format::input(&path)
            .map_err(|e| VideoError::Backend(format!("Failed to open input: {}", e)))?;

        let input = ictx
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or_else(|| VideoError::Backend("No video stream found".to_string()))?;

        let stream_index = input.index();
        let context = ffmpeg::codec::context::Context::from_parameters(input.parameters())
            .map_err(|e| VideoError::Backend(format!("Failed to get codec context: {}", e)))?;

        let decoder = context
            .decoder()
            .video()
            .map_err(|e| VideoError::Backend(format!("Failed to get decoder: {}", e)))?;

        let width = decoder.width();
        let height = decoder.height();

        let scaler = ffmpeg::software::scaling::context::Context::get(
            decoder.format(),
            width,
            height,
            ffmpeg::format::Pixel::GRAY8,
            width,
            height,
            ffmpeg::software::scaling::flag::BILINEAR,
        )
        .map_err(|e| VideoError::Backend(format!("Failed to initialize scaler: {}", e)))?;

        Ok(Self {
            ictx,
            decoder,
            stream_index,
            width,
            height,
            scaler,
            current_frame: None,
        })
    }
}

impl VideoCapture for NativeFfmpegCapture {
    fn is_opened(&self) -> bool {
        true
    }

    fn grab(&mut self) -> Result<()> {
        // Read packets until a frame is decoded
        for (stream, packet) in self.ictx.packets() {
            if stream.index() == self.stream_index {
                self.decoder
                    .send_packet(&packet)
                    .map_err(|e| VideoError::CaptureFailed(format!("Send packet failed: {}", e)))?;

                let mut decoded = ffmpeg::util::frame::Video::empty();
                if self.decoder.receive_frame(&mut decoded).is_ok() {
                    let mut gray_frame = ffmpeg::util::frame::Video::new(
                        ffmpeg::format::Pixel::GRAY8,
                        self.width,
                        self.height,
                    );

                    self.scaler
                        .run(&decoded, &mut gray_frame)
                        .map_err(|e| VideoError::CaptureFailed(format!("Scaling failed: {}", e)))?;

                    let data = gray_frame.data(0).to_vec();
                    let stride = gray_frame.stride(0) as usize; // Stride is usually i32/isize, cast to usize
                    let width = self.width as usize;

                    let final_data = if stride == width {
                        data
                    } else {
                        // Copy row by row removing padding
                        let height = self.height as usize;
                        let mut unpadded = Vec::with_capacity(width * height);
                        for i in 0..height {
                            let start = i * stride;
                            let end = start + width;
                            if end <= data.len() {
                                unpadded.extend_from_slice(&data[start..end]);
                            }
                        }
                        unpadded
                    };

                    let img = GrayImage::from_raw(self.width, self.height, final_data).ok_or_else(
                        || VideoError::CaptureFailed("Image creation failed".to_string()),
                    )?;

                    self.current_frame = Some(img);
                    return Ok(());
                }
            }
        }

        // If we reach here, EOF or no more frames
        // Flush decoder
        self.decoder.send_eof().ok();
        let mut decoded = ffmpeg::util::frame::Video::empty();
        if self.decoder.receive_frame(&mut decoded).is_ok() {
            let mut gray_frame = ffmpeg::util::frame::Video::new(
                ffmpeg::format::Pixel::GRAY8,
                self.width,
                self.height,
            );

            self.scaler
                .run(&decoded, &mut gray_frame)
                .map_err(|e| VideoError::CaptureFailed(format!("Scaling failed: {}", e)))?;

            let data = gray_frame.data(0).to_vec();
            let stride = gray_frame.stride(0) as usize;
            let width = self.width as usize;
            let final_data = if stride == width {
                data
            } else {
                let height = self.height as usize;
                let mut unpadded = Vec::with_capacity(width * height);
                for i in 0..height {
                    let start = i * stride;
                    let end = start + width;
                    if end <= data.len() {
                        unpadded.extend_from_slice(&data[start..end]);
                    }
                }
                unpadded
            };

            let img = GrayImage::from_raw(self.width, self.height, final_data)
                .ok_or_else(|| VideoError::CaptureFailed("Image creation failed".to_string()))?;
            self.current_frame = Some(img);
            return Ok(());
        }

        Err(VideoError::CaptureFailed("End of stream".to_string()))
    }

    fn retrieve(&mut self) -> Result<GrayImage> {
        self.current_frame
            .clone()
            .ok_or_else(|| VideoError::CaptureFailed("No frame grabbed".to_string()))
    }
}
