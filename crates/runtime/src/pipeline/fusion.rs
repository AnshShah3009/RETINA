use super::{BufferId, PipelineNode};
use crate::Result;
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FusionPattern {
    ConvThreshold,
    ConvConv,
    ThresholdNms,
    GaussianThreshold,
    SobelCanny,
    Custom(Vec<String>),
}

#[derive(Debug, Clone)]
pub struct FusedKernel {
    pub name: String,
    pub original_nodes: Vec<usize>,
    pub inputs: Vec<BufferId>,
    pub outputs: Vec<BufferId>,
    pub combined_params: Vec<u8>,
    pub shader_source: Option<String>,
}

pub struct KernelFuser {
    enabled: bool,
    max_fusion_depth: usize,
    fusible_kernels: HashSet<String>,
}

impl KernelFuser {
    pub fn new() -> Self {
        let mut fusible = HashSet::new();
        fusible.insert("conv2d".into());
        fusible.insert("threshold".into());
        fusible.insert("gaussian_blur".into());
        fusible.insert("sobel".into());
        fusible.insert("nms".into());
        fusible.insert("canny".into());

        Self {
            enabled: true,
            max_fusion_depth: 3,
            fusible_kernels: fusible,
        }
    }

    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    pub fn with_max_fusion_depth(mut self, depth: usize) -> Self {
        self.max_fusion_depth = depth;
        self
    }

    pub fn is_fusible(&self, node: &PipelineNode) -> bool {
        match node {
            PipelineNode::Kernel { name, .. } => self.fusible_kernels.contains(name),
            _ => false,
        }
    }

    pub fn detect_pattern(&self, nodes: &[&PipelineNode]) -> Option<FusionPattern> {
        if nodes.is_empty() || nodes.len() > self.max_fusion_depth {
            return None;
        }

        let names: Vec<&str> = nodes
            .iter()
            .filter_map(|n| {
                if let PipelineNode::Kernel { name, .. } = n {
                    Some(name.as_str())
                } else {
                    None
                }
            })
            .collect();

        if names.len() != nodes.len() {
            return None;
        }

        match names.as_slice() {
            ["conv2d", "threshold"] => Some(FusionPattern::ConvThreshold),
            ["gaussian_blur", "threshold"] => Some(FusionPattern::GaussianThreshold),
            ["sobel", "canny"] => Some(FusionPattern::SobelCanny),
            ["threshold", "nms"] => Some(FusionPattern::ThresholdNms),
            ["conv2d", "conv2d"] => Some(FusionPattern::ConvConv),
            _ => None,
        }
    }

    pub fn try_fuse(&self, nodes: &[PipelineNode]) -> Result<Vec<FusedKernel>> {
        if !self.enabled {
            return Ok(Vec::new());
        }

        let mut fused_kernels = Vec::new();
        let mut fused_indices = HashSet::new();
        let mut i = 0;

        while i < nodes.len() {
            if fused_indices.contains(&i) {
                i += 1;
                continue;
            }

            let mut best_fusion: Option<(usize, FusedKernel)> = None;

            for depth in (2..=self.max_fusion_depth.min(nodes.len() - i)).rev() {
                let window: Vec<&PipelineNode> = (i..i + depth).map(|idx| &nodes[idx]).collect();

                if let Some(pattern) = self.detect_pattern(&window) {
                    let all_fusible = window.iter().all(|n| self.is_fusible(n));

                    if all_fusible {
                        if let Ok(fused) = self.create_fused_kernel(&window, i, &pattern) {
                            best_fusion = Some((depth, fused));
                            break;
                        }
                    }
                }
            }

            if let Some((depth, fused)) = best_fusion {
                for j in i..i + depth {
                    fused_indices.insert(j);
                }
                fused_kernels.push(fused);
                i += depth;
            } else {
                i += 1;
            }
        }

        Ok(fused_kernels)
    }

    fn create_fused_kernel(
        &self,
        nodes: &[&PipelineNode],
        start_idx: usize,
        pattern: &FusionPattern,
    ) -> Result<FusedKernel> {
        let original_indices: Vec<usize> = (start_idx..start_idx + nodes.len()).collect();

        let mut all_inputs = Vec::new();
        let mut all_outputs = Vec::new();
        let mut combined_params = Vec::new();

        let intermediate_buffers: HashSet<BufferId> = nodes
            .windows(2)
            .filter_map(|w| {
                if let (
                    PipelineNode::Kernel { outputs: out1, .. },
                    PipelineNode::Kernel { inputs: in2, .. },
                ) = (&w[0], &w[1])
                {
                    let intermediate: HashSet<BufferId> =
                        out1.iter().filter(|o| in2.contains(o)).cloned().collect();
                    Some(intermediate)
                } else {
                    None
                }
            })
            .flatten()
            .collect();

        for node in nodes {
            if let PipelineNode::Kernel {
                inputs,
                outputs,
                params,
                ..
            } = node
            {
                for input in inputs {
                    if !intermediate_buffers.contains(input) && !all_inputs.contains(input) {
                        all_inputs.push(*input);
                    }
                }
                for output in outputs {
                    if !intermediate_buffers.contains(output) && !all_outputs.contains(output) {
                        all_outputs.push(*output);
                    }
                }
                combined_params.extend_from_slice(params);
            }
        }

        let fused_name = match pattern {
            FusionPattern::ConvThreshold => "fused_conv_threshold",
            FusionPattern::ConvConv => "fused_conv_conv",
            FusionPattern::ThresholdNms => "fused_threshold_nms",
            FusionPattern::GaussianThreshold => "fused_gaussian_threshold",
            FusionPattern::SobelCanny => "fused_sobel_canny",
            FusionPattern::Custom(names) => {
                return Err(crate::Error::RuntimeError(format!(
                    "Custom fusion pattern not yet implemented: {:?}",
                    names
                )));
            }
        };

        let shader_source = Some(generate_fused_wgsl(pattern));

        Ok(FusedKernel {
            name: fused_name.to_string(),
            original_nodes: original_indices,
            inputs: all_inputs,
            outputs: all_outputs,
            combined_params,
            shader_source,
        })
    }

    pub fn optimize(&self, nodes: Vec<PipelineNode>) -> Result<Vec<PipelineNode>> {
        let fused = self.try_fuse(&nodes)?;

        if fused.is_empty() {
            return Ok(nodes);
        }

        let mut optimized = Vec::new();
        let skip_indices: HashSet<usize> = fused
            .iter()
            .flat_map(|f| f.original_nodes.iter().copied())
            .collect();

        let mut fused_iter = fused.into_iter().peekable();

        for (i, node) in nodes.into_iter().enumerate() {
            if skip_indices.contains(&i) {
                continue;
            }

            if let Some(f) = fused_iter.peek() {
                if f.original_nodes.first() == Some(&i) {
                    let fused_kernel = fused_iter.next().unwrap();
                    optimized.push(PipelineNode::Kernel {
                        name: fused_kernel.name,
                        inputs: fused_kernel.inputs,
                        outputs: fused_kernel.outputs,
                        params: fused_kernel.combined_params,
                    });
                }
            }

            let first_fused_idx = fused_iter.peek().and_then(|f| f.original_nodes.first());
            if first_fused_idx != Some(&i) && !skip_indices.contains(&i) {
                optimized.push(node);
            }
        }

        for fused_kernel in fused_iter {
            let last_fused = optimized.last().and_then(|n| {
                if let PipelineNode::Kernel { name, .. } = n {
                    Some(name.clone())
                } else {
                    None
                }
            });

            if last_fused.as_deref() != Some(fused_kernel.name.as_str()) {
                optimized.push(PipelineNode::Kernel {
                    name: fused_kernel.name,
                    inputs: fused_kernel.inputs,
                    outputs: fused_kernel.outputs,
                    params: fused_kernel.combined_params,
                });
            }
        }

        Ok(optimized)
    }
}

impl Default for KernelFuser {
    fn default() -> Self {
        Self::new()
    }
}

fn generate_fused_wgsl(pattern: &FusionPattern) -> String {
    match pattern {
        FusionPattern::GaussianThreshold => r#"
// Fused Gaussian Blur + Threshold
struct Params {
    width: u32,
    height: u32,
    sigma: f32,
    kernel_size: u32,
    thresh: u32,
    max_value: u32,
    thresh_type: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> temp_data: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_data: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

fn get_pixel(x: i32, y: i32) -> f32 {
    let w = i32(params.width);
    let h = i32(params.height);
    let ix = clamp(x, 0, w - 1);
    let iy = clamp(y, 0, h - 1);
    let idx = u32(iy * w + ix);
    let combined = input_data[idx / 4u];
    return f32((combined >> ((idx % 4u) * 8u)) & 0xFFu);
}

fn apply_gaussian() {
    let w = params.width;
    let h = params.height;
    let size = (params.kernel_size - 1u) / 2u;
    let count = (w + 3u) / 4u * h;

    for (var i = 0u; i < count; i++) {
        let y = i / ((w + 3u) / 4u);
        let x_base = i % ((w + 3u) / 4u);
        var sum = 0.0;
        var weight_sum = 0.0;

        for (var dy = -i32(size); dy <= i32(size); dy++) {
            for (var dx = -i32(size); dx <= i32(size); dx++) {
                let px = i32(x_base * 4u) + dx;
                let py = i32(y) + dy;
                let val = get_pixel(px, py);
                let dist_sq = f32(dx * dx + dy * dy);
                let weight = exp(-dist_sq / (2.0 * params.sigma * params.sigma));
                sum = sum + val * weight;
                weight_sum = weight_sum + weight;
            }
        }

        let avg = u32(sum / weight_sum);
        temp_data[i] = (avg << 24u) | (avg << 16u) | (avg << 8u) | avg;
    }
}

fn apply_thresh(val: u32) -> u32 {
    if (params.thresh_type == 0u) {
        if (val > params.thresh) { return params.max_value; } else { return 0u; }
    } else if (params.thresh_type == 2u) {
        return min(val, params.thresh);
    }
    return val;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let count = (params.width + 3u) / 4u * params.height;

    if (idx < 1u) { apply_gaussian(); }

    if (idx < count) {
        let combined = temp_data[idx];
        var res_combined = 0u;
        for (var i = 0u; i < 4u; i++) {
            let pixel_idx = idx * 4u + i;
            if (pixel_idx >= params.width * params.height) { break; }
            let val = (combined >> (i * 8u)) & 0xFFu;
            res_combined = res_combined | ((apply_thresh(val) & 0xFFu) << (i * 8u));
        }
        output_data[idx] = res_combined;
    }
}
"#
        .to_string(),

        FusionPattern::ThresholdNms => r#"
// Fused Threshold + NMS
struct Params {
    width: u32,
    height: u32,
    thresh: u32,
    max_value: u32,
    nms_radius: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> temp_data: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_data: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

fn apply_thresh(val: u32) -> u32 {
    if (val > params.thresh) { return params.max_value; }
    return 0u;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let count = (params.width + 3u) / 4u * params.height;
    let w = (params.width + 3u) / 4u;

    if (idx < count) {
        let combined = input_data[idx];
        var threshed = 0u;
        for (var i = 0u; i < 4u; i++) {
            let pixel_idx = idx * 4u + i;
            if (pixel_idx >= params.width * params.height) { break; }
            let val = (combined >> (i * 8u)) & 0xFFu;
            threshed = threshed | ((apply_thresh(val) & 0xFFu) << (i * 8u));
        }
        temp_data[idx] = threshed;
    }

    workgroupBarrier();

    if (idx < count) {
        let y = idx / w;
        let x_base = idx % w;
        var is_max = true;
        let my_val = temp_data[idx];

        for (var dy = -i32(params.nms_radius); dy <= i32(params.nms_radius); dy++) {
            for (var dx = -i32(params.nms_radius); dx <= i32(params.nms_radius); dx++) {
                if (dx == 0 && dy == 0) { continue; }
                let nx = i32(x_base) + dx;
                let ny = i32(y) + dy;
                if (nx < 0 || nx >= i32(w) || ny < 0 || ny >= i32(params.height)) { continue; }
                let nidx = u32(ny) * w + u32(nx);
                if (temp_data[nidx] > my_val) { is_max = false; }
            }
        }

        output_data[idx] = if (is_max) { threshed } else { 0u };
    }
}
"#
        .to_string(),

        FusionPattern::SobelCanny => r#"
// Fused Sobel + Canny Edge Detection
struct Params {
    width: u32,
    height: u32,
    low_thresh: u32,
    high_thresh: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> gx_data: array<u32>;
@group(0) @binding(2) var<storage, read_write> gy_data: array<u32>;
@group(0) @binding(3) var<storage, read_write> mag_data: array<u32>;
@group(0) @binding(4) var<storage, read_write> output_data: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;

fn get_pixel(x: i32, y: i32) -> f32 {
    let w = i32(params.width);
    let h = i32(params.height);
    let ix = clamp(x, 0, w - 1);
    let iy = clamp(y, 0, h - 1);
    let idx = u32(iy * w + ix);
    let combined = input_data[idx / 4u];
    return f32((combined >> ((idx % 4u) * 8u)) & 0xFFu);
}

fn compute_sobel() {
    let w = params.width;
    let h = params.height;

    for (var y = 0u; y < h; y++) {
        for (var x = 0u; x < (w + 3u) / 4u; x++) {
            let idx = x + y * ((w + 3u) / 4u);
            var res_gx = 0u;
            var res_gy = 0u;

            for (var i = 0u; i < 4u; i++) {
                let px = i32(x * 4u + i);
                if (u32(px) >= w) { break; }

                let p00 = get_pixel(px - 1, i32(y) - 1);
                let p01 = get_pixel(px, i32(y) - 1);
                let p02 = get_pixel(px + 1, i32(y) - 1);
                let p10 = get_pixel(px - 1, i32(y));
                let p12 = get_pixel(px + 1, i32(y));
                let p20 = get_pixel(px - 1, i32(y) + 1);
                let p21 = get_pixel(px, i32(y) + 1);
                let p22 = get_pixel(px + 1, i32(y) + 1);

                let gx = (p02 + 2.0 * p12 + p22) - (p00 + 2.0 * p10 + p20);
                let gy = (p20 + 2.0 * p21 + p22) - (p00 + 2.0 * p01 + p02);

                res_gx = res_gx | ((u32(clamp(gx + 128.0, 0.0, 255.0)) & 0xFFu) << (i * 8u));
                res_gy = res_gy | ((u32(clamp(gy + 128.0, 0.0, 255.0)) & 0xFFu) << (i * 8u));
            }

            gx_data[idx] = res_gx;
            gy_data[idx] = res_gy;
        }
    }
}

fn compute_magnitude() {
    let w = params.width;
    let h = params.height;

    for (var y = 0u; y < h; y++) {
        for (var x = 0u; x < (w + 3u) / 4u; x++) {
            let idx = x + y * ((w + 3u) / 4u);
            let combined_gx = gx_data[idx];
            let combined_gy = gy_data[idx];
            var res_mag = 0u;

            for (var i = 0u; i < 4u; i++) {
                let gx = f32((combined_gx >> (i * 8u)) & 0xFFu) - 128.0;
                let gy = f32((combined_gy >> (i * 8u)) & 0xFFu) - 128.0;
                let mag = u32(sqrt(gx * gx + gy * gy));
                res_mag = res_mag | ((min(mag, 255u) & 0xFFu) << (i * 8u));
            }

            mag_data[idx] = res_mag;
        }
    }
}

fn apply_canny() {
    let w = params.width;
    let h = params.height;

    for (var y = 0u; y < h; y++) {
        for (var x = 0u; x < (w + 3u) / 4u; x++) {
            let idx = x + y * ((w + 3u) / 4u);
            let combined = mag_data[idx];
            var res = 0u;

            for (var i = 0u; i < 4u; i++) {
                let pixel_idx = idx * 4u + i;
                if (pixel_idx >= w * h) { break; }

                let mag = (combined >> (i * 8u)) & 0xFFu;

                if (mag < params.low_thresh) {
                    res = res | (0u << (i * 8u));
                } else if (mag >= params.high_thresh) {
                    res = res | (255u << (i * 8u));
                } else {
                    res = res | (128u << (i * 8u));
                }
            }

            output_data[idx] = res;
        }
    }
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x == 0u && global_id.y == 0u) {
        compute_sobel();
        compute_magnitude();
        apply_canny();
    }
}
"#
        .to_string(),

        FusionPattern::ConvThreshold => r#"
// Fused Convolution + Threshold
struct Params {
    width: u32,
    height: u32,
    kernel_size: u32,
    kernel_offset: u32,
    thresh: u32,
    max_value: u32,
    thresh_type: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn get_pixel(x: i32, y: i32) -> f32 {
    let w = i32(params.width);
    let h = i32(params.height);
    let ix = clamp(x, 0, w - 1);
    let iy = clamp(y, 0, h - 1);
    let idx = u32(iy * w + ix);
    let combined = input_data[idx / 4u];
    return f32((combined >> ((idx % 4u) * 8u)) & 0xFFu);
}

fn apply_conv(x: i32, y: i32) -> f32 {
    let ksize = i32(params.kernel_size);
    let half = ksize / 2;
    var sum = 0.0;
    for (var ky = 0; ky < ksize; ky++) {
        for (var kx = 0; kx < ksize; kx++) {
            let px = x + kx - half;
            let py = y + ky - half;
            let val = get_pixel(px, py);
            let kernel_idx = u32(ky * ksize + kx + i32(params.kernel_offset));
            let kernel_val = f32(kernel_idx % 9u) - 4.0;
            sum = sum + val * kernel_val;
        }
    }
    return sum;
}

fn apply_thresh(val: u32) -> u32 {
    if (params.thresh_type == 0u) {
        if (val > params.thresh) { return params.max_value; } else { return 0u; }
    }
    return val;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let w4 = (params.width + 3u) / 4u;
    let y = idx / w4;
    let x_base = idx % w4;
    let total = w4 * params.height;

    if (idx >= total) { return; }

    var res_combined = 0u;
    for (var i = 0u; i < 4u; i++) {
        let x = i32(x_base * 4u + i);
        if (u32(x) >= params.width) { break; }
        let conv_val = apply_conv(x, i32(y));
        let conv_u8 = u32(clamp(conv_val + 128.0, 0.0, 255.0));
        res_combined = res_combined | ((apply_thresh(conv_u8) & 0xFFu) << (i * 8u));
    }

    output_data[idx] = res_combined;
}
"#
        .to_string(),

        FusionPattern::ConvConv => r#"
// Fused Convolution + Convolution
struct Params {
    width: u32,
    height: u32,
    kernel1_offset: u32,
    kernel2_offset: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> temp_data: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_data: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

fn get_pixel(x: i32, y: i32) -> f32 {
    let w = i32(params.width);
    let h = i32(params.height);
    let ix = clamp(x, 0, w - 1);
    let iy = clamp(y, 0, h - 1);
    let idx = u32(iy * w + ix);
    let combined = input_data[idx / 4u];
    return f32((combined >> ((idx % 4u) * 8u)) & 0xFFu);
}

fn apply_conv_at(x: i32, y: i32, kernel_offset: u32) -> f32 {
    let ksize = 3i32;
    let half = ksize / 2;
    var sum = 0.0;
    for (var ky = 0; ky < ksize; ky++) {
        for (var kx = 0; kx < ksize; kx++) {
            let px = x + kx - half;
            let py = y + ky - half;
            let val = get_pixel(px, py);
            let kernel_idx = u32(ky * ksize + kx) + kernel_offset;
            let kernel_val = f32(kernel_idx % 9u) - 4.0;
            sum = sum + val * kernel_val;
        }
    }
    return sum;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let w4 = (params.width + 3u) / 4u;
    let y = idx / w4;
    let x_base = idx % w4;
    let total = w4 * params.height;

    if (idx >= total) { return; }

    // First convolution
    var temp_combined = 0u;
    for (var i = 0u; i < 4u; i++) {
        let x = i32(x_base * 4u + i);
        if (u32(x) >= params.width) { break; }
        let val = apply_conv_at(x, i32(y), params.kernel1_offset);
        temp_combined = temp_combined | ((u32(clamp(val + 128.0, 0.0, 255.0)) & 0xFFu) << (i * 8u));
    }
    temp_data[idx] = temp_combined;

    workgroupBarrier();

    // Second convolution
    var res_combined = 0u;
    for (var i = 0u; i < 4u; i++) {
        let x = i32(x_base * 4u + i);
        if (u32(x) >= params.width) { break; }
        let val = apply_conv_at(x, i32(y), params.kernel2_offset);
        res_combined = res_combined | ((u32(clamp(val + 128.0, 0.0, 255.0)) & 0xFFu) << (i * 8u));
    }

    output_data[idx] = res_combined;
}
"#
        .to_string(),

        FusionPattern::Custom(_) => r#"
// Custom fused kernel (placeholder)
struct Params { width: u32, height: u32 }

@group(0) @binding(0) var<storage, read> input_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let count = (params.width + 3u) / 4u * params.height;
    if (idx < count) {
        output_data[idx] = input_data[idx];
    }
}
"#
        .to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_detection() {
        let fuser = KernelFuser::new();

        let node1 = PipelineNode::Kernel {
            name: "conv2d".into(),
            inputs: vec![BufferId(0)],
            outputs: vec![BufferId(1)],
            params: vec![],
        };
        let node2 = PipelineNode::Kernel {
            name: "threshold".into(),
            inputs: vec![BufferId(1)],
            outputs: vec![BufferId(2)],
            params: vec![],
        };

        let nodes: Vec<&PipelineNode> = vec![&node1, &node2];

        let pattern = fuser.detect_pattern(&nodes);
        assert_eq!(pattern, Some(FusionPattern::ConvThreshold));
    }

    #[test]
    fn test_no_fusion_for_non_fusible() {
        let fuser = KernelFuser::new();

        let nodes = vec![PipelineNode::Barrier, PipelineNode::Barrier];

        let fused = fuser.try_fuse(&nodes).unwrap();
        assert!(fused.is_empty());
    }

    #[test]
    fn test_fusion_creates_fused_kernel() {
        let fuser = KernelFuser::new();

        let nodes = vec![
            PipelineNode::Kernel {
                name: "conv2d".into(),
                inputs: vec![BufferId(0)],
                outputs: vec![BufferId(1)],
                params: vec![1, 2, 3],
            },
            PipelineNode::Kernel {
                name: "threshold".into(),
                inputs: vec![BufferId(1)],
                outputs: vec![BufferId(2)],
                params: vec![4, 5],
            },
        ];

        let fused = fuser.try_fuse(&nodes).unwrap();
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].name, "fused_conv_threshold");
        assert_eq!(fused[0].original_nodes, vec![0, 1]);
        assert_eq!(fused[0].inputs, vec![BufferId(0)]);
        assert_eq!(fused[0].outputs, vec![BufferId(2)]);
    }
}
