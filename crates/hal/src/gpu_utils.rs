//! Shared GPU utilities for memory budgeting and configuration
//!
//! This module provides common helpers for GPU resource management
//! across different CV modules (stereo, features, video, etc.)

use crate::Error;
use std::env;

/// Parse GPU memory budget from environment variable RUSTCV_GPU_MAX_BYTES
///
/// Supports formats:
/// - Plain bytes: `134217728`
/// - With suffix: `512MB`, `2GB`, `256KB`
///
/// # Example
///
/// ```ignore
/// export RUSTCV_GPU_MAX_BYTES=1GB
/// let budget = read_gpu_max_bytes_from_env()?;
/// assert_eq!(budget, Some(1024 * 1024 * 1024));
/// ```
pub fn read_gpu_max_bytes_from_env() -> crate::Result<Option<usize>> {
    let raw = match env::var("RUSTCV_GPU_MAX_BYTES") {
        Ok(v) => v,
        Err(env::VarError::NotPresent) => return Ok(None),
        Err(e) => {
            return Err(Error::MemoryError(format!(
                "Failed to read RUSTCV_GPU_MAX_BYTES: {e}"
            )))
        }
    };

    let parsed = parse_bytes_with_suffix(&raw)?;
    if parsed == 0 {
        return Err(Error::MemoryError(
            "RUSTCV_GPU_MAX_BYTES must be >= 1".to_string(),
        ));
    }

    Ok(Some(parsed))
}

/// Parse a byte size string with optional suffix (KB, MB, GB)
///
/// # Supported formats
/// - Plain: `1024`
/// - KB: `512KB`
/// - MB: `256MB`
/// - GB: `2GB`
/// - Underscores for readability: `1_000_000_000`
pub fn parse_bytes_with_suffix(raw: &str) -> crate::Result<usize> {
    let s = raw.trim();
    if s.is_empty() {
        return Err(Error::MemoryError(
            "GPU memory size cannot be empty".to_string(),
        ));
    }

    let upper = s.to_ascii_uppercase().replace('_', "");
    let (number_part, multiplier): (&str, usize) = if let Some(v) = upper.strip_suffix("KB") {
        (v, 1024)
    } else if let Some(v) = upper.strip_suffix("MB") {
        (v, 1024 * 1024)
    } else if let Some(v) = upper.strip_suffix("GB") {
        (v, 1024 * 1024 * 1024)
    } else if let Some(v) = upper.strip_suffix('B') {
        (v, 1)
    } else {
        (upper.as_str(), 1)
    };

    let base: usize = number_part.parse().map_err(|_| {
        Error::MemoryError(format!(
            "Memory size must be like '134217728', '512MB', or '2GB'; got '{raw}'"
        ))
    })?;

    base.checked_mul(multiplier).ok_or_else(|| {
        Error::MemoryError(format!(
            "Memory size value '{raw}' is too large (would overflow)"
        ))
    })
}

/// Calculate memory usage for GPU buffers
///
/// # Example
///
/// ```ignore
/// let image_width = 1920u32;
/// let image_height = 1080u32;
/// let bytes_per_pixel = 1; // grayscale
/// let memory = estimate_image_buffer_size(image_width, image_height, bytes_per_pixel);
/// ```
pub fn estimate_image_buffer_size(width: u32, height: u32, bytes_per_pixel: u32) -> usize {
    (width as usize) * (height as usize) * (bytes_per_pixel as usize)
}

/// Check if operation fits within GPU memory budget
///
/// Returns true if `required_bytes <= budget`, false otherwise
pub fn fits_in_budget(required_bytes: usize, budget: Option<usize>) -> bool {
    match budget {
        Some(b) => required_bytes <= b,
        None => true, // No limit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_bytes_plain() {
        assert_eq!(parse_bytes_with_suffix("1024").unwrap(), 1024);
        // parse_bytes_with_suffix allows 0, but read_gpu_max_bytes_from_env rejects it
        assert_eq!(parse_bytes_with_suffix("0").unwrap(), 0);
    }

    #[test]
    fn test_parse_bytes_with_suffix() {
        assert_eq!(parse_bytes_with_suffix("512KB").unwrap(), 512 * 1024);
        assert_eq!(parse_bytes_with_suffix("256MB").unwrap(), 256 * 1024 * 1024);
        assert_eq!(
            parse_bytes_with_suffix("2GB").unwrap(),
            2 * 1024 * 1024 * 1024
        );
    }

    #[test]
    fn test_parse_bytes_case_insensitive() {
        assert_eq!(
            parse_bytes_with_suffix("1mb").unwrap(),
            parse_bytes_with_suffix("1MB").unwrap()
        );
    }

    #[test]
    fn test_parse_bytes_with_underscores() {
        assert_eq!(parse_bytes_with_suffix("1_000_000").unwrap(), 1_000_000);
    }

    #[test]
    fn test_estimate_image_buffer_size() {
        let size = estimate_image_buffer_size(1920, 1080, 1);
        assert_eq!(size, 1920 * 1080);
    }

    #[test]
    fn test_fits_in_budget() {
        assert!(fits_in_budget(100, Some(200)));
        assert!(!fits_in_budget(200, Some(100)));
        assert!(fits_in_budget(1000, None)); // No limit
    }
}
