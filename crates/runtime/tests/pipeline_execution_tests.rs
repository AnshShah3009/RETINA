//! Tests for pipeline kernel execution.

use cv_runtime::pipeline::Pipeline;
use std::sync::Arc;

#[test]
fn test_pipeline_creation() {
    let _pipeline = Pipeline::new();
    // Pipeline should be creatable
    assert!(true, "Pipeline created");
}

#[test]
fn test_pipeline_build_empty() {
    let pipeline = Pipeline::new();

    // Build empty pipeline should succeed
    let result = pipeline.build();
    assert!(result.is_ok());
}

#[test]
fn test_pipeline_execute() {
    let pipeline = Pipeline::new().build().expect("build should succeed");

    // Execute on a runner should return a result
    let runner = cv_runtime::orchestrator::RuntimeRunner::Sync(cv_hal::DeviceId(0));
    let result = pipeline.execute(&runner);
    assert!(result.is_ok());
}

#[test]
fn test_pipeline_node_types() {
    // Test that PipelineNode enum exists
    let _node = cv_runtime::pipeline::PipelineNode::Barrier;
    assert!(true, "PipelineNode enum exists");
}

#[test]
fn test_pipeline_node_debug() {
    let node = cv_runtime::pipeline::PipelineNode::Barrier;
    let debug = format!("{:?}", node);
    assert_eq!(debug, "Barrier");
}

#[test]
fn test_pipeline_node_debug_kernel() {
    let node = cv_runtime::pipeline::PipelineNode::Kernel {
        name: "test".to_string(),
        inputs: vec![],
        outputs: vec![],
        params: vec![],
    };
    let debug = format!("{:?}", node);
    assert!(debug.contains("test"));
}

#[test]
fn test_pipeline_node_debug_cpu_op() {
    let node = cv_runtime::pipeline::PipelineNode::CpuOp {
        inputs: vec![],
        outputs: vec![],
        op: Arc::new(|_| vec![]),
    };
    let debug = format!("{:?}", node);
    assert!(debug.contains("CpuOp"));
}

#[test]
fn test_pipeline_node_equality() {
    let node1 = cv_runtime::pipeline::PipelineNode::Barrier;
    let node2 = cv_runtime::pipeline::PipelineNode::Barrier;
    // We can't use assert_eq! without PartialEq, but we can compare debug output
    let debug1 = format!("{:?}", node1);
    let debug2 = format!("{:?}", node2);
    assert_eq!(debug1, debug2);
}

#[test]
fn test_pipeline_node_clone() {
    let node = cv_runtime::pipeline::PipelineNode::Kernel {
        name: "test".to_string(),
        inputs: vec![],
        outputs: vec![],
        params: vec![],
    };
    let cloned = node.clone();
    let debug1 = format!("{:?}", node);
    let debug2 = format!("{:?}", cloned);
    assert_eq!(debug1, debug2);
}
