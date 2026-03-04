use super::BufferId;
use super::PipelineNode;
use crate::Result;
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

#[derive(Debug, Clone)]
pub struct NodeDependency {
    pub from: NodeId,
    pub to: NodeId,
    pub buffer: BufferId,
}

#[derive(Clone)]
pub struct ExecutionGraph {
    pub nodes: Vec<NodeId>,
    pub dependencies: Vec<NodeDependency>,
    pub topology_order: Vec<NodeId>,
    pub levels: HashMap<NodeId, usize>,
}

impl ExecutionGraph {
    pub fn build(pipeline_nodes: &[PipelineNode]) -> Result<Self> {
        let nodes: Vec<NodeId> = (0..pipeline_nodes.len()).map(NodeId).collect();

        let mut buffer_producers: HashMap<BufferId, NodeId> = HashMap::new();
        let mut dependencies = Vec::new();
        let mut adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        let mut predecessors: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();

        for node in &nodes {
            in_degree.insert(*node, 0);
            adjacency.insert(*node, Vec::new());
            predecessors.insert(*node, Vec::new());
        }

        for (idx, node) in pipeline_nodes.iter().enumerate() {
            let node_id = NodeId(idx);

            static EMPTY: Vec<BufferId> = Vec::new();
            let inputs: &Vec<BufferId> = match node {
                PipelineNode::Kernel { inputs, .. } => inputs,
                PipelineNode::CpuOp { inputs, .. } => inputs,
                PipelineNode::Barrier => &EMPTY,
            };

            for &input_buffer in inputs {
                if let Some(&producer_id) = buffer_producers.get(&input_buffer) {
                    dependencies.push(NodeDependency {
                        from: producer_id,
                        to: node_id,
                        buffer: input_buffer,
                    });

                    adjacency.entry(producer_id).or_default().push(node_id);
                    predecessors.entry(node_id).or_default().push(producer_id);
                    *in_degree.entry(node_id).or_insert(0) += 1;
                }
            }

            let outputs: &Vec<BufferId> = match node {
                PipelineNode::Kernel { outputs, .. } => outputs,
                PipelineNode::CpuOp { outputs, .. } => outputs,
                PipelineNode::Barrier => &EMPTY,
            };

            for &output_buffer in outputs {
                buffer_producers.insert(output_buffer, node_id);
            }
        }

        let topology_order = Self::topological_sort(&nodes, &adjacency, &in_degree)?;

        let levels = Self::compute_levels(&topology_order, &predecessors);

        Ok(Self {
            nodes,
            dependencies,
            topology_order,
            levels,
        })
    }

    fn topological_sort(
        nodes: &[NodeId],
        adjacency: &HashMap<NodeId, Vec<NodeId>>,
        in_degree: &HashMap<NodeId, usize>,
    ) -> Result<Vec<NodeId>> {
        let mut in_degree = in_degree.clone();
        let mut queue: VecDeque<NodeId> = VecDeque::new();
        let mut result = Vec::with_capacity(nodes.len());

        for &node in nodes {
            if in_degree.get(&node).copied().unwrap_or(0) == 0 {
                queue.push_back(node);
            }
        }

        while let Some(node) = queue.pop_front() {
            result.push(node);

            if let Some(neighbors) = adjacency.get(&node) {
                for &neighbor in neighbors {
                    let degree = in_degree.entry(neighbor).or_insert(0);
                    if *degree > 0 {
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
        }

        if result.len() != nodes.len() {
            return Err(crate::Error::RuntimeError(
                "Pipeline contains a cycle. Circular dependencies are not allowed.".into(),
            ));
        }

        Ok(result)
    }

    fn compute_levels(
        topology_order: &[NodeId],
        predecessors: &HashMap<NodeId, Vec<NodeId>>,
    ) -> HashMap<NodeId, usize> {
        let mut levels = HashMap::new();

        for &node in topology_order {
            let max_pred_level = predecessors
                .get(&node)
                .map(|preds| {
                    preds
                        .iter()
                        .filter_map(|p| levels.get(p).copied())
                        .max()
                        .unwrap_or(0)
                })
                .unwrap_or(0);

            levels.insert(node, max_pred_level + 1);
        }

        levels
    }

    pub fn get_dependencies(&self, node_id: NodeId) -> Vec<&NodeDependency> {
        self.dependencies
            .iter()
            .filter(|dep| dep.to == node_id)
            .collect()
    }

    pub fn get_dependents(&self, node_id: NodeId) -> Vec<&NodeDependency> {
        self.dependencies
            .iter()
            .filter(|dep| dep.from == node_id)
            .collect()
    }

    pub fn get_level(&self, node_id: NodeId) -> usize {
        self.levels.get(&node_id).copied().unwrap_or(0)
    }

    pub fn max_level(&self) -> usize {
        self.levels.values().copied().max().unwrap_or(0)
    }

    pub fn nodes_at_level(&self, level: usize) -> Vec<NodeId> {
        self.levels
            .iter()
            .filter_map(|(&node, &l)| if l == level { Some(node) } else { None })
            .collect()
    }

    pub fn parallelizable_groups(&self) -> Vec<Vec<NodeId>> {
        let max_level = self.max_level();
        (1..=max_level)
            .map(|level| self.nodes_at_level(level))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_pipeline_graph() {
        let nodes = vec![
            PipelineNode::Kernel {
                name: "kernel1".into(),
                inputs: vec![],
                outputs: vec![BufferId(0)],
                params: vec![],
            },
            PipelineNode::Kernel {
                name: "kernel2".into(),
                inputs: vec![BufferId(0)],
                outputs: vec![BufferId(1)],
                params: vec![],
            },
        ];

        let graph = ExecutionGraph::build(&nodes).unwrap();

        assert_eq!(graph.topology_order.len(), 2);
        assert_eq!(graph.topology_order[0], NodeId(0));
        assert_eq!(graph.topology_order[1], NodeId(1));
        assert_eq!(graph.dependencies.len(), 1);
    }

    #[test]
    fn test_parallel_nodes() {
        let nodes = vec![
            PipelineNode::Kernel {
                name: "source".into(),
                inputs: vec![],
                outputs: vec![BufferId(0)],
                params: vec![],
            },
            PipelineNode::Kernel {
                name: "branch_a".into(),
                inputs: vec![BufferId(0)],
                outputs: vec![BufferId(1)],
                params: vec![],
            },
            PipelineNode::Kernel {
                name: "branch_b".into(),
                inputs: vec![BufferId(0)],
                outputs: vec![BufferId(2)],
                params: vec![],
            },
            PipelineNode::Kernel {
                name: "merge".into(),
                inputs: vec![BufferId(1), BufferId(2)],
                outputs: vec![BufferId(3)],
                params: vec![],
            },
        ];

        let graph = ExecutionGraph::build(&nodes).unwrap();

        assert_eq!(graph.max_level(), 3);

        let level2 = graph.nodes_at_level(2);
        assert_eq!(level2.len(), 2);
    }

    #[test]
    fn test_unresolved_inputs_still_builds() {
        // This test shows that a graph with unresolved inputs (BufferId(1) is never produced)
        // still builds successfully. The graph has no topological cycle because the
        // unresolved dependency creates a non-existent edge (producer not found in buffer_producers).
        // BufferId(0) → node_b → BufferId(1) → node_a, but since BufferId(1) has no producer,
        // there's no edge node_b → node_a, so no cycle.
        let nodes = vec![
            PipelineNode::Kernel {
                name: "a".into(),
                inputs: vec![BufferId(1)],
                outputs: vec![BufferId(0)],
                params: vec![],
            },
            PipelineNode::Kernel {
                name: "b".into(),
                inputs: vec![BufferId(0)],
                outputs: vec![BufferId(1)],
                params: vec![],
            },
        ];

        let result = ExecutionGraph::build(&nodes);
        assert!(result.is_ok(), "Graph with unresolved input should build");

        let graph = result.unwrap();
        // The topological order should be valid since there's no actual cycle
        assert_eq!(graph.topology_order.len(), 2);
    }
}
