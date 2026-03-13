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

        Ok(FusedKernel {
            name: fused_name.to_string(),
            original_nodes: original_indices,
            inputs: all_inputs,
            outputs: all_outputs,
            combined_params,
            shader_source: None,
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
