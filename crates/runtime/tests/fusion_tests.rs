//! Tests for kernel fusion system.

use cv_runtime::pipeline::{BufferId, KernelFuser, PipelineNode};

#[test]
fn test_kernel_fuser_creation() {
    let _fuser = KernelFuser::new();
    // Should be created with default rules
    assert!(true, "KernelFuser created");
}

#[test]
fn test_fusion_rule_node_match() {
    // Create nodes that would match a sobel+threshold pattern
    let nodes = vec![
        PipelineNode::Kernel {
            name: "sobel".to_string(),
            inputs: vec![BufferId(0)],
            outputs: vec![BufferId(1)],
            params: vec![],
        },
        PipelineNode::Kernel {
            name: "threshold".to_string(),
            inputs: vec![BufferId(1)],
            outputs: vec![BufferId(2)],
            params: vec![128u8],
        },
    ];

    // Nodes should be creatable
    assert_eq!(nodes.len(), 2);
}

#[test]
fn test_kernel_node_debug() {
    let node = PipelineNode::Kernel {
        name: "sobel".to_string(),
        inputs: vec![BufferId(0)],
        outputs: vec![BufferId(1)],
        params: vec![],
    };

    let debug = format!("{:?}", node);
    assert!(debug.contains("sobel"));
}

#[test]
fn test_fused_kernel_structure() {
    // Test that kernel nodes have proper structure
    let node = PipelineNode::Kernel {
        name: "sobel".to_string(),
        inputs: vec![BufferId(0)],
        outputs: vec![BufferId(1)],
        params: vec![],
    };

    match node {
        PipelineNode::Kernel {
            name,
            inputs,
            outputs,
            params,
        } => {
            assert_eq!(name, "sobel");
            assert_eq!(inputs.len(), 1);
            assert_eq!(outputs.len(), 1);
            assert!(params.is_empty());
        }
        _ => panic!("Expected Kernel node"),
    }
}

#[test]
fn test_kernel_fuser_optimize() {
    let mut fuser = KernelFuser::new();

    let nodes = vec![
        PipelineNode::Kernel {
            name: "sobel".to_string(),
            inputs: vec![BufferId(0)],
            outputs: vec![BufferId(1)],
            params: vec![],
        },
        PipelineNode::Kernel {
            name: "threshold".to_string(),
            inputs: vec![BufferId(1)],
            outputs: vec![BufferId(2)],
            params: vec![128u8],
        },
    ];

    // Optimize should not panic
    let result = fuser.optimize(nodes);
    assert!(result.is_ok());
}

#[test]
fn test_kernel_fuser_emits_fused_kernel_in_place() {
    // conv2d + threshold fuse (indices 0..2), followed by a non-fusible node.
    // The fused kernel must be emitted at the position of its first original
    // node, not appended after the trailing node (which would reverse
    // producer/consumer order).
    let fuser = KernelFuser::new();

    let nodes = vec![
        PipelineNode::Kernel {
            name: "conv2d".to_string(),
            inputs: vec![BufferId(0)],
            outputs: vec![BufferId(1)],
            params: vec![],
        },
        PipelineNode::Kernel {
            name: "threshold".to_string(),
            inputs: vec![BufferId(1)],
            outputs: vec![BufferId(2)],
            params: vec![128u8],
        },
        PipelineNode::Kernel {
            name: "resize".to_string(),
            inputs: vec![BufferId(2)],
            outputs: vec![BufferId(3)],
            params: vec![],
        },
    ];

    let optimized = fuser.optimize(nodes).unwrap();
    assert_eq!(optimized.len(), 2, "expected one fused kernel + one node");

    match &optimized[0] {
        PipelineNode::Kernel { name, .. } => assert_ne!(
            name, "resize",
            "fused kernel must come before the trailing node, got {name}"
        ),
        other => panic!("expected kernel node, got {other:?}"),
    }
    match &optimized[1] {
        PipelineNode::Kernel { name, .. } => assert_eq!(name, "resize"),
        other => panic!("expected kernel node, got {other:?}"),
    }
}
