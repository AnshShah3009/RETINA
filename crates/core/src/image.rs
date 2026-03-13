use image::{GrayImage, ImageBuffer, Luma, Rgb, RgbImage, RgbaImage};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelType {
    U8,
    U16,
    F32,
    F64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    Gray,
    Rgb,
    Rgba,
    Bgr,
    Hsv,
    Lab,
}

pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
    pub pixel_type: PixelType,
    pub channels: u8,
}

impl ImageInfo {
    pub fn new(width: u32, height: u32, channels: u8, pixel_type: PixelType) -> Self {
        Self {
            width,
            height,
            channels,
            pixel_type,
        }
    }

    pub fn is_gray(&self) -> bool {
        self.channels == 1
    }

    pub fn is_color(&self) -> bool {
        self.channels >= 3
    }

    pub fn stride(&self) -> usize {
        let bytes_per_pixel = match self.pixel_type {
            PixelType::U8 => 1,
            PixelType::U16 => 2,
            PixelType::F32 => 4,
            PixelType::F64 => 8,
        };
        (self.width as usize * self.channels as usize) * bytes_per_pixel
    }
}

pub trait CvImage: Clone {
    fn width(&self) -> u32;
    fn height(&self) -> u32;
    fn as_slice(&self) -> &[u8];
    fn as_mut_slice(&mut self) -> &mut [u8];
}

impl CvImage for GrayImage {
    fn width(&self) -> u32 {
        self.width()
    }
    fn height(&self) -> u32 {
        self.height()
    }
    fn as_slice(&self) -> &[u8] {
        self.as_raw()
    }
    fn as_mut_slice(&mut self) -> &mut [u8] {
        self.as_mut()
    }
}

impl CvImage for RgbImage {
    fn width(&self) -> u32 {
        self.width()
    }
    fn height(&self) -> u32 {
        self.height()
    }
    fn as_slice(&self) -> &[u8] {
        self.as_raw()
    }
    fn as_mut_slice(&mut self) -> &mut [u8] {
        self.as_mut()
    }
}

impl CvImage for RgbaImage {
    fn width(&self) -> u32 {
        self.width()
    }
    fn height(&self) -> u32 {
        self.height()
    }
    fn as_slice(&self) -> &[u8] {
        self.as_raw()
    }
    fn as_mut_slice(&mut self) -> &mut [u8] {
        self.as_mut()
    }
}

pub type ImageRef<'a, T> = &'a ImageBuffer<T, Vec<u8>>;

pub fn create_gray_image(width: u32, height: u32) -> GrayImage {
    ImageBuffer::new(width, height)
}

pub fn create_rgb_image(width: u32, height: u32) -> RgbImage {
    ImageBuffer::new(width, height)
}

pub fn create_rgba_image(width: u32, height: u32) -> RgbaImage {
    ImageBuffer::new(width, height)
}

pub fn get_pixel_gray(img: &GrayImage, x: u32, y: u32) -> u8 {
    img.get_pixel(x, y)[0]
}

pub fn set_pixel_gray(img: &mut GrayImage, x: u32, y: u32, value: u8) {
    img.put_pixel(x, y, Luma([value]));
}

pub fn get_pixel_rgb(img: &RgbImage, x: u32, y: u32) -> [u8; 3] {
    let p = img.get_pixel(x, y);
    [p[0], p[1], p[2]]
}

pub fn set_pixel_rgb(img: &mut RgbImage, x: u32, y: u32, rgb: [u8; 3]) {
    img.put_pixel(x, y, Rgb(rgb));
}

pub fn convert_gray_to_rgb(gray: &GrayImage) -> RgbImage {
    let (w, h) = gray.dimensions();
    let mut rgb = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let val = gray.get_pixel(x, y)[0];
            rgb.put_pixel(x, y, Rgb([val, val, val]));
        }
    }
    rgb
}

pub fn convert_rgb_to_gray(rgb: &RgbImage) -> GrayImage {
    let (w, h) = rgb.dimensions();
    let mut gray = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let p = rgb.get_pixel(x, y);
            let gray_val = (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) as u8;
            gray.put_pixel(x, y, Luma([gray_val]));
        }
    }
    gray
}
