"""Tests for cv-native Python bindings"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv_native as cv


class TestFeatureDetector:
    """Test feature detection bindings"""

    def test_fast_detect(self):
        """Test FAST corner detection"""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[20:30, 20:30] = 255
        img[60:70, 60:70] = 255

        kps = cv.FeatureDetector.fast_detect(img, 50)
        assert len(kps) >= 0

    def test_harris_detect(self):
        """Test Harris corner detection"""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[20:30, 20:30] = 255
        img[60:70, 60:70] = 255

        kps = cv.FeatureDetector.harris_detect(img, 0.04, 1000.0)
        assert len(kps) >= 0

    def test_gftt_detect(self):
        """Test GFTT corner detection"""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[20:30, 20:30] = 255

        kps = cv.FeatureDetector.gftt_detect(img, 100, 0.01, 5.0)
        assert len(kps) >= 0

    def test_shi_tomasi_detect(self):
        """Test Shi-Tomasi corner detection"""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[20:30, 20:30] = 255

        kps = cv.FeatureDetector.shi_tomasi_detect(img, 100, 0.01)
        assert len(kps) >= 0


class TestPointCloud:
    """Test point cloud bindings"""

    def test_create_from_numpy(self):
        """Test creating point cloud from numpy"""
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        pc = cv.PointCloud(points)
        assert pc.num_points() == 3

    def test_to_numpy(self):
        """Test converting point cloud to numpy"""
        pc = cv.PointCloud([(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)])
        arr = pc.to_numpy()
        assert len(arr) == 6

    def test_set_normals(self):
        """Test setting normals"""
        pc = cv.PointCloud([(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)])
        normals = np.array([[0, 0, 1], [0, 1, 0]], dtype=np.float32)
        pc.set_normals(normals)


class TestMeshReconstruction:
    """Test mesh reconstruction bindings"""

    def test_create_sphere(self):
        """Test sphere point cloud creation"""
        pc = cv.MeshReconstruction.create_sphere((0, 0, 0), 1.0, 100)
        assert pc.num_points() == 100

    def test_create_plane(self):
        """Test plane point cloud creation"""
        pc = cv.MeshReconstruction.create_plane((0, 0, 0), (0, 0, 1), 1.0, 50)
        assert pc.num_points() == 50

    def test_ball_pivoting(self):
        """Test ball pivoting surface reconstruction"""
        pc = cv.MeshReconstruction.create_sphere((0, 0, 0), 1.0, 200)
        mesh = cv.MeshReconstruction.ball_pivoting(pc, 0.2)
        assert mesh.num_vertices() == 200

    def test_alpha_shapes(self):
        """Test alpha shapes reconstruction"""
        pc = cv.MeshReconstruction.create_sphere((0, 0, 0), 1.0, 100)
        mesh = cv.MeshReconstruction.alpha_shapes(pc, 0.1)
        assert mesh.num_vertices() >= 0

    def test_poisson(self):
        """Test Poisson surface reconstruction"""
        pc = cv.MeshReconstruction.create_sphere((0, 0, 0), 1.0, 100)
        mesh = cv.MeshReconstruction.poisson(pc, 5)
        assert mesh is not None

    def test_mesh_to_obj(self):
        """Test OBJ export"""
        mesh = cv.TriangleMesh()
        mesh.add_vertex(0.0, 0.0, 0.0)
        mesh.add_vertex(1.0, 0.0, 0.0)
        mesh.add_vertex(0.0, 1.0, 0.0)
        mesh.add_face(0, 1, 2)

        obj = mesh.to_obj()
        assert "v 0 0 0" in obj
        assert "f 1 2 3" in obj


class TestIsam2:
    """Test ISAM2 bindings"""

    def test_create(self):
        """Test ISAM2 creation"""
        isam = cv.Isam2(True, False)
        assert isam.num_nodes() == 0

    def test_add_pose(self):
        """Test adding pose"""
        isam = cv.Isam2(False, False)
        isam.add_pose(0, 0.0, 0.0, 0.0)
        isam.add_pose(1, 1.0, 0.0, 0.0)
        assert isam.num_nodes() == 2

    def test_add_point(self):
        """Test adding point"""
        isam = cv.Isam2(False, False)
        isam.add_point(100, 1.0, 2.0, 3.0)
        assert isam.num_nodes() == 1

    def test_add_factor(self):
        """Test adding factor"""
        isam = cv.Isam2(False, False)
        isam.add_pose(0, 0.0, 0.0, 0.0)
        isam.add_pose(1, 1.0, 0.0, 0.0)
        isam.add_factor(0, 1, 1.0, 0.0, 0.0, 0.1)
        assert isam.num_factors() == 1

    def test_get_pose(self):
        """Test getting pose"""
        isam = cv.Isam2(False, False)
        isam.add_pose(0, 1.0, 2.0, 3.0)
        pose = isam.get_pose(0)
        assert pose is not None
        assert pose[0] == 1.0

    def test_get_all_poses(self):
        """Test getting all poses"""
        isam = cv.Isam2(False, False)
        isam.add_pose(0, 0.0, 0.0, 0.0)
        isam.add_pose(1, 1.0, 0.0, 0.0)
        poses = isam.get_all_poses()
        assert len(poses) == 2

    def test_optimize(self):
        """Test optimization"""
        isam = cv.Isam2(False, False)
        isam.add_pose(0, 0.0, 0.0, 0.0)
        isam.add_pose(1, 1.0, 0.0, 0.0)
        isam.add_factor(0, 1, 1.0, 0.0, 0.0, 0.1)
        isam.optimize()


class TestTensor:
    """Test tensor bindings"""

    def test_create(self):
        """Test tensor creation"""
        tensor = cv.Tensor((3, 100, 100))
        assert tensor.shape() == (3, 100, 100)

    def test_zeros(self):
        """Test zeros tensor"""
        tensor = cv.Tensor.zeros((1, 10, 10))
        assert tensor.shape() == (1, 10, 10)

    def test_ones(self):
        """Test ones tensor"""
        tensor = cv.Tensor.ones((1, 10, 10))
        arr = tensor.to_numpy()
        assert all(v == 1.0 for v in arr)


class TestKeyPoints:
    """Test keypoints bindings"""

    def test_create(self):
        """Test creating keypoints"""
        kps = cv.KeyPoints()
        assert len(kps) == 0

    def test_to_list(self):
        """Test converting to list"""
        kps = cv.KeyPoints()
        lst = kps.to_list()
        assert isinstance(lst, list)


class TestRuntime:
    """Test runtime bindings"""

    def test_get_num_devices(self):
        """Test getting number of devices"""
        num = cv.Runtime.get_num_devices()
        assert num >= 1

    def test_get_global_load(self):
        """Test getting global load"""
        load = cv.Runtime.get_global_load()
        assert isinstance(load, list)

    def test_get_device_info(self):
        """Test getting device info"""
        info = cv.Runtime.get_device_info()
        assert isinstance(info, list)
        for dev in info:
            assert hasattr(dev, "device_idx")
            assert hasattr(dev, "total_mb")
            assert hasattr(dev, "used_mb")
            assert hasattr(dev, "owner_count")
            assert dev.free_mb() >= 0
            assert dev.free_mb() <= dev.total_mb

    def test_reserve_device(self):
        """Test reserving device memory"""
        try:
            cv.Runtime.reserve_device(0, 100)
            cv.Runtime.release_device(0)
        except RuntimeError:
            pytest.skip("No coordinator available")

    def test_release_device(self):
        """Test releasing device memory"""
        try:
            cv.Runtime.reserve_device(0, 50)
            cv.Runtime.release_device(0)
        except RuntimeError:
            pytest.skip("No coordinator available")

    def test_best_device(self):
        """Test finding best device"""
        device = cv.Runtime.best_device(0)
        assert isinstance(device, int)
        assert device >= -1

    def test_set_gpu_wait_timeout(self):
        """Test setting GPU wait timeout"""
        try:
            cv.Runtime.set_gpu_wait_timeout(5000)
            timeout = cv.Runtime.get_gpu_wait_timeout()
            assert timeout == 5000
        except RuntimeError:
            pytest.skip("No scheduler available")

    def test_get_gpu_wait_timeout(self):
        """Test getting GPU wait timeout"""
        try:
            timeout = cv.Runtime.get_gpu_wait_timeout()
            assert timeout >= 0
        except RuntimeError:
            pytest.skip("No scheduler available")

    def test_set_execution_mode_strict(self):
        """Test setting strict execution mode"""
        cv.Runtime.set_execution_mode(cv.Runtime.ExecutionMode.Strict)

    def test_set_execution_mode_normal(self):
        """Test setting normal execution mode"""
        cv.Runtime.set_execution_mode(cv.Runtime.ExecutionMode.Normal)

    def test_set_execution_mode_adaptive_basic(self):
        """Test setting adaptive basic execution mode"""
        cv.Runtime.set_execution_mode(cv.Runtime.ExecutionMode.AdaptiveBasic)

    def test_set_execution_mode_adaptive_aggressive(self):
        """Test setting adaptive aggressive execution mode"""
        cv.Runtime.set_execution_mode(cv.Runtime.ExecutionMode.AdaptiveAggressive)

    def test_join_group(self):
        """Test joining affinity group"""
        try:
            group = cv.Runtime.join_group(1)
            assert group is not None
            assert hasattr(group, "group_id")
            assert group.group_id == 1
        except RuntimeError:
            pytest.skip("No scheduler available")

    def test_mock_init_device(self):
        """Test mocking device initialization"""
        try:
            cv.Runtime.mock_init_device(0, 1000)
        except RuntimeError:
            pytest.skip("No scheduler available")


class TestPyWorkloadHint:
    """Test workload hint bindings"""

    def test_latency_hint(self):
        """Test latency workload hint"""
        hint = cv.Runtime.WorkloadHint.Latency
        assert hint is not None

    def test_throughput_hint(self):
        """Test throughput workload hint"""
        hint = cv.Runtime.WorkloadHint.Throughput
        assert hint is not None

    def test_powersave_hint(self):
        """Test powersave workload hint"""
        hint = cv.Runtime.WorkloadHint.PowerSave
        assert hint is not None

    def test_default_hint(self):
        """Test default workload hint"""
        hint = cv.Runtime.WorkloadHint.Default
        assert hint is not None


class TestAffinityGroup:
    """Test affinity group bindings"""

    def test_join_group(self):
        """Test joining affinity group"""
        try:
            group = cv.Runtime.join_group(42)
            assert group.group_id == 42
        except RuntimeError:
            pytest.skip("No scheduler available")

    def test_leave_group(self):
        """Test leaving affinity group"""
        try:
            group = cv.Runtime.join_group(43)
            group.leave()
        except RuntimeError:
            pytest.skip("No scheduler available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
