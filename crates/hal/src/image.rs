use std::sync::Arc;
use wgpu::{Texture, TextureFormat, TextureView};

/// A GPU-resident image
#[derive(Debug)]
pub struct GpuImage {
    pub texture: Arc<Texture>,
    pub view: Arc<TextureView>,
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
}

impl GpuImage {
    pub fn new(texture: Texture, width: u32, height: u32, format: TextureFormat) -> Self {
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            texture: Arc::new(texture),
            view: Arc::new(view),
            width,
            height,
            format,
        }
    }
}
