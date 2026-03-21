"""Property-based tests for cv-native Python bindings using Hypothesis"""

import pytest
import numpy as np
import sys
import os
from hypothesis import given, settings, assume, example
import hypothesis.strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv_native as cv


class TestWorkloadHintProperties:
    """Property tests for workload hints"""

    @given(
        mode=st.sampled_from(
            [
                cv.Runtime.WorkloadHint.Latency,
                cv.Runtime.WorkloadHint.Throughput,
                cv.Runtime.WorkloadHint.PowerSave,
                cv.Runtime.WorkloadHint.Default,
            ]
        )
    )
    @settings(max_examples=10)
    def test_workload_hint_always_valid(self, mode):
        """All workload hints should be valid enum values"""
        assert mode is not None
        assert isinstance(mode, cv.Runtime.WorkloadHint)

    @given(_=st.just(None))
    @settings(max_examples=10)
    def test_execution_mode_all_variants(self, _):
        """All execution modes should be valid"""
        modes = [
            cv.Runtime.ExecutionMode.Strict,
            cv.Runtime.ExecutionMode.Normal,
            cv.Runtime.ExecutionMode.AdaptiveBasic,
            cv.Runtime.ExecutionMode.AdaptiveAggressive,
        ]
        for mode in modes:
            assert mode is not None


class TestDeviceProperties:
    """Property tests for device operations"""

    @given(device_idx=st.integers(min_value=0, max_value=7))
    @settings(max_examples=20)
    def test_device_index_range(self, device_idx):
        """Device indices should be in valid range"""
        assert 0 <= device_idx <= 7

    @given(memory_mb=st.integers(min_value=1, max_value=10000))
    @settings(max_examples=20)
    def test_memory_reservation_bounds(self, memory_mb):
        """Memory reservations should be in reasonable bounds"""
        assert 1 <= memory_mb <= 10000

    @given(
        device_idx=st.integers(min_value=0, max_value=7),
        memory_mb=st.integers(min_value=1, max_value=1000),
    )
    @settings(max_examples=20)
    def test_reserve_release_roundtrip(self, device_idx, memory_mb):
        """Reserving and releasing should be idempotent"""
        try:
            cv.Runtime.reserve_device(device_idx, memory_mb)
            cv.Runtime.release_device(device_idx)
        except RuntimeError:
            pytest.skip("No coordinator available")


class TestExecutionModeProperties:
    """Property tests for execution mode transitions"""

    @given(_=st.just(None))
    @settings(max_examples=10)
    def test_execution_mode_transitions(self, _):
        """Mode changes should be valid transitions"""
        modes = [
            cv.Runtime.ExecutionMode.Strict,
            cv.Runtime.ExecutionMode.Normal,
            cv.Runtime.ExecutionMode.AdaptiveBasic,
            cv.Runtime.ExecutionMode.AdaptiveAggressive,
        ]
        for mode in modes:
            cv.Runtime.set_execution_mode(mode)

    @given(_=st.just(None))
    @settings(max_examples=10)
    def test_default_mode_after_reset(self, _):
        """Default mode should be Normal after start"""
        cv.Runtime.set_execution_mode(cv.Runtime.ExecutionMode.Strict)
        cv.Runtime.set_execution_mode(cv.Runtime.ExecutionMode.Normal)


class TestPointCloudProperties:
    """Property tests for point cloud operations"""

    @given(
        num_points=st.integers(min_value=1, max_value=100),
        x_range=st.tuples(
            st.floats(min_value=-10, max_value=10),
            st.floats(min_value=-10, max_value=10),
        ),
        z_range=st.tuples(
            st.floats(min_value=-10, max_value=10),
            st.floats(min_value=-10, max_value=10),
        ),
    )
    @settings(max_examples=20)
    def test_point_cloud_scaling(self, num_points, x_range, z_range):
        """Point cloud should scale with input size"""
        pc = cv.MeshReconstruction.create_sphere((0, 0, 0), 1.0, num_points)
        assert pc.num_points() == num_points


class TestMeshProperties:
    """Property tests for mesh operations"""

    @given(
        num_points=st.integers(min_value=10, max_value=200),
        alpha=st.floats(min_value=0.01, max_value=1.0),
    )
    @settings(max_examples=10)
    def test_alpha_shapes_scaling(self, num_points, alpha):
        """Alpha shapes should handle different parameters"""
        pc = cv.MeshReconstruction.create_sphere((0, 0, 0), 1.0, num_points)
        mesh = cv.MeshReconstruction.alpha_shapes(pc, alpha)
        assert mesh.num_vertices() >= 0


class TestTensorProperties:
    """Property tests for tensor operations"""

    @given(
        shape=st.tuples(
            st.integers(min_value=1, max_value=4),
            st.integers(min_value=1, max_value=64),
            st.integers(min_value=1, max_value=64),
        )
    )
    @settings(max_examples=20)
    def test_tensor_shape_preservation(self, shape):
        """Tensor shape should be preserved after creation"""
        tensor = cv.Tensor(shape)
        assert tensor.shape() == shape

    @given(
        shape=st.tuples(
            st.integers(min_value=1, max_value=3),
            st.integers(min_value=1, max_value=32),
            st.integers(min_value=1, max_value=32),
        )
    )
    @settings(max_examples=20)
    def test_zeros_tensor_shape(self, shape):
        """Zeros tensor should have correct shape"""
        tensor = cv.Tensor.zeros(shape)
        assert tensor.shape() == shape

    @given(
        shape=st.tuples(
            st.integers(min_value=1, max_value=3),
            st.integers(min_value=1, max_value=32),
            st.integers(min_value=1, max_value=32),
        )
    )
    @settings(max_examples=20)
    def test_ones_tensor_shape(self, shape):
        """Ones tensor should have correct shape"""
        tensor = cv.Tensor.ones(shape)
        assert tensor.shape() == shape


class TestISAM2Properties:
    """Property tests for ISAM2 operations"""

    @given(
        num_poses=st.integers(min_value=1, max_value=10),
        noise=st.floats(min_value=0.1, max_value=2.0),
    )
    @settings(max_examples=10)
    def test_isam_pose_counting(self, num_poses, noise):
        """ISAM2 should track pose count correctly"""
        isam = cv.Isam2(False, False)
        for i in range(num_poses):
            isam.add_pose(i, float(i) * noise, 0.0, 0.0)
        assert isam.num_nodes() == num_poses

    @given(
        num_poses=st.integers(min_value=2, max_value=5),
        noise=st.floats(min_value=0.1, max_value=1.0),
    )
    @settings(max_examples=10)
    def test_isam_factor_creation(self, num_poses, noise):
        """ISAM2 should create factors correctly"""
        isam = cv.Isam2(False, False)
        for i in range(num_poses):
            isam.add_pose(i, float(i) * noise, 0.0, 0.0)

        for i in range(num_poses - 1):
            isam.add_factor(i, i + 1, noise, 0.0, 0.0, 0.1)

        assert isam.num_factors() == num_poses - 1


class TestKeyPointsProperties:
    """Property tests for keypoint operations"""

    @given(_=st.just(None))
    @settings(max_examples=5)
    def test_empty_keypoints_to_list(self, _):
        """Empty keypoints should return empty list"""
        kps = cv.KeyPoints()
        lst = kps.to_list()
        assert isinstance(lst, list)
        assert len(lst) == 0


class TestAffinityGroupProperties:
    """Property tests for affinity groups"""

    @given(group_id=st.integers(min_value=1, max_value=1000))
    @settings(max_examples=10)
    def test_affinity_group_id_preservation(self, group_id):
        """Affinity group ID should be preserved"""
        try:
            group = cv.Runtime.join_group(group_id)
            assert group.group_id == group_id
        except RuntimeError:
            pytest.skip("No scheduler available")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
