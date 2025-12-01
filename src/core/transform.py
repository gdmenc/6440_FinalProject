# Helper functions for Euler/Quaternion/Matrix math
import numpy as np
from numpy.typing import NDArray

Mat4x4 = NDArray[np.float32]

def get_translation_matrix(x: float, y: float, z: float) -> Mat4x4:
    """
    Returns a 4x4 Translation Matrix.
    """
    return np.array(
        [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]], dtype=np.float32
    )


def get_rotation_matrix_x(angle_degrees: float) -> Mat4x4:
    """
    Returns a 4x4 Rotation Matrix along the x-axis.
    """
    radians = np.radians(angle_degrees)
    c, s = np.cos(radians), np.sin(radians)
    return np.array(
        [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=np.float32
    )


def get_rotation_matrix_y(angle_degrees: float) -> Mat4x4:
    """
    Returns a 4x4 Rotation Matrix along the y-axis.
    """
    radians = np.radians(angle_degrees)
    c, s = np.cos(radians), np.sin(radians)
    return np.array(
        [[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], dtype=np.float32
    )


def get_rotation_matrix_z(angle_degrees: float) -> Mat4x4:
    """
    Returns a 4x4 Rotation Matrix along the z-axis.
    """
    radians = np.radians(angle_degrees)
    c, s = np.cos(radians), np.sin(radians)
    return np.array(
        [[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
    )


def distance_squared(pos1: NDArray[np.float32], pos2: NDArray[np.float32]) -> float:
    return np.sum((pos1 - pos2) ** 2)
