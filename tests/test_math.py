# Unit tests for matrix math
import unittest
import numpy as np
from src.core.transform import (
    get_translation_matrix,
    get_rotation_matrix_x,
    get_rotation_matrix_y,
    get_rotation_matrix_z,
    distance_squared
)


class TestTranslationMatrix(unittest.TestCase):
    """Tests for translation matrix generation."""
    
    def test_identity_translation(self):
        """Test that (0,0,0) produces identity-like matrix."""
        T = get_translation_matrix(0, 0, 0)
        expected = np.eye(4, dtype=np.float32)
        np.testing.assert_array_almost_equal(T, expected)
    
    def test_positive_translation(self):
        """Test translation with positive values."""
        T = get_translation_matrix(1.0, 2.0, 3.0)
        self.assertEqual(T[0, 3], 1.0)
        self.assertEqual(T[1, 3], 2.0)
        self.assertEqual(T[2, 3], 3.0)
        self.assertEqual(T[3, 3], 1.0)
    
    def test_negative_translation(self):
        """Test translation with negative values."""
        T = get_translation_matrix(-5.0, -10.0, -15.0)
        self.assertEqual(T[0, 3], -5.0)
        self.assertEqual(T[1, 3], -10.0)
        self.assertEqual(T[2, 3], -15.0)
    
    def test_matrix_shape(self):
        """Test that output is 4x4."""
        T = get_translation_matrix(1, 2, 3)
        self.assertEqual(T.shape, (4, 4))
        self.assertEqual(T.dtype, np.float32)


class TestRotationMatrixX(unittest.TestCase):
    """Tests for X-axis rotation matrices."""
    
    def test_zero_rotation(self):
        """Test that 0 degrees produces identity matrix."""
        R = get_rotation_matrix_x(0)
        expected = np.eye(4, dtype=np.float32)
        np.testing.assert_array_almost_equal(R, expected)
    
    def test_90_degree_rotation(self):
        """Test 90 degree rotation around X-axis."""
        R = get_rotation_matrix_x(90)
        # Rotating (0,1,0) around X by 90° should give (0,0,1)
        point = np.array([0, 1, 0, 1], dtype=np.float32)
        result = R @ point
        expected = np.array([0, 0, 1, 1], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
    
    def test_180_degree_rotation(self):
        """Test 180 degree rotation around X-axis."""
        R = get_rotation_matrix_x(180)
        # Rotating (0,1,0) around X by 180° should give (0,-1,0)
        point = np.array([0, 1, 0, 1], dtype=np.float32)
        result = R @ point
        expected = np.array([0, -1, 0, 1], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
    
    def test_360_degree_rotation(self):
        """Test 360 degree rotation returns to identity."""
        R = get_rotation_matrix_x(360)
        expected = np.eye(4, dtype=np.float32)
        np.testing.assert_array_almost_equal(R, expected, decimal=5)
    
    def test_negative_rotation(self):
        """Test that negative angles work correctly."""
        R_pos = get_rotation_matrix_x(45)
        R_neg = get_rotation_matrix_x(-45)
        # R(-θ) should be the transpose (inverse) of R(θ) for rotation matrices
        np.testing.assert_array_almost_equal(R_neg, R_pos.T, decimal=5)
    
    def test_matrix_shape(self):
        """Test that output is 4x4."""
        R = get_rotation_matrix_x(45)
        self.assertEqual(R.shape, (4, 4))
        self.assertEqual(R.dtype, np.float32)


class TestRotationMatrixY(unittest.TestCase):
    """Tests for Y-axis rotation matrices."""
    
    def test_zero_rotation(self):
        """Test that 0 degrees produces identity matrix."""
        R = get_rotation_matrix_y(0)
        expected = np.eye(4, dtype=np.float32)
        np.testing.assert_array_almost_equal(R, expected)
    
    def test_90_degree_rotation(self):
        """Test 90 degree rotation around Y-axis."""
        R = get_rotation_matrix_y(90)
        # Rotating (1,0,0) around Y by 90° should give (0,0,-1)
        point = np.array([1, 0, 0, 1], dtype=np.float32)
        result = R @ point
        expected = np.array([0, 0, -1, 1], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
    
    def test_180_degree_rotation(self):
        """Test 180 degree rotation around Y-axis."""
        R = get_rotation_matrix_y(180)
        # Rotating (1,0,0) around Y by 180° should give (-1,0,0)
        point = np.array([1, 0, 0, 1], dtype=np.float32)
        result = R @ point
        expected = np.array([-1, 0, 0, 1], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
    
    def test_360_degree_rotation(self):
        """Test 360 degree rotation returns to identity."""
        R = get_rotation_matrix_y(360)
        expected = np.eye(4, dtype=np.float32)
        np.testing.assert_array_almost_equal(R, expected, decimal=5)
    
    def test_matrix_shape(self):
        """Test that output is 4x4."""
        R = get_rotation_matrix_y(45)
        self.assertEqual(R.shape, (4, 4))
        self.assertEqual(R.dtype, np.float32)


class TestRotationMatrixZ(unittest.TestCase):
    """Tests for Z-axis rotation matrices."""
    
    def test_zero_rotation(self):
        """Test that 0 degrees produces identity matrix."""
        R = get_rotation_matrix_z(0)
        expected = np.eye(4, dtype=np.float32)
        np.testing.assert_array_almost_equal(R, expected)
    
    def test_90_degree_rotation(self):
        """Test 90 degree rotation around Z-axis."""
        R = get_rotation_matrix_z(90)
        # Rotating (1,0,0) around Z by 90° should give (0,1,0)
        point = np.array([1, 0, 0, 1], dtype=np.float32)
        result = R @ point
        expected = np.array([0, 1, 0, 1], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
    
    def test_180_degree_rotation(self):
        """Test 180 degree rotation around Z-axis."""
        R = get_rotation_matrix_z(180)
        # Rotating (1,0,0) around Z by 180° should give (-1,0,0)
        point = np.array([1, 0, 0, 1], dtype=np.float32)
        result = R @ point
        expected = np.array([-1, 0, 0, 1], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
    
    def test_360_degree_rotation(self):
        """Test 360 degree rotation returns to identity."""
        R = get_rotation_matrix_z(360)
        expected = np.eye(4, dtype=np.float32)
        np.testing.assert_array_almost_equal(R, expected, decimal=5)
    
    def test_matrix_shape(self):
        """Test that output is 4x4."""
        R = get_rotation_matrix_z(45)
        self.assertEqual(R.shape, (4, 4))
        self.assertEqual(R.dtype, np.float32)


class TestDistanceSquared(unittest.TestCase):
    """Tests for squared distance calculation."""
    
    def test_same_point(self):
        """Test distance from point to itself is zero."""
        pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        dist_sq = distance_squared(pos, pos)
        self.assertAlmostEqual(dist_sq, 0.0, places=5)
    
    def test_unit_distance_x(self):
        """Test distance along x-axis."""
        pos1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        pos2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        dist_sq = distance_squared(pos1, pos2)
        self.assertAlmostEqual(dist_sq, 1.0, places=5)
    
    def test_unit_distance_y(self):
        """Test distance along y-axis."""
        pos1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        pos2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        dist_sq = distance_squared(pos1, pos2)
        self.assertAlmostEqual(dist_sq, 1.0, places=5)
    
    def test_unit_distance_z(self):
        """Test distance along z-axis."""
        pos1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        pos2 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        dist_sq = distance_squared(pos1, pos2)
        self.assertAlmostEqual(dist_sq, 1.0, places=5)
    
    def test_3d_distance(self):
        """Test distance in 3D space (3-4-5 triangle)."""
        pos1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        pos2 = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        dist_sq = distance_squared(pos1, pos2)
        # 3^2 + 4^2 = 9 + 16 = 25
        self.assertAlmostEqual(dist_sq, 25.0, places=5)
    
    def test_negative_coordinates(self):
        """Test distance with negative coordinates."""
        pos1 = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        pos2 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        dist_sq = distance_squared(pos1, pos2)
        # (2^2 + 2^2 + 2^2) = 12
        self.assertAlmostEqual(dist_sq, 12.0, places=5)
    
    def test_commutative(self):
        """Test that distance(a,b) == distance(b,a)."""
        pos1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        pos2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        dist1 = distance_squared(pos1, pos2)
        dist2 = distance_squared(pos2, pos1)
        self.assertAlmostEqual(dist1, dist2, places=5)


if __name__ == '__main__':
    unittest.main()
