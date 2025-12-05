from __future__ import annotations
from typing import List, Optional, Dict, Tuple
from numpy.typing import NDArray
import numpy as np
from src.core.transform import (
    Mat4x4,
    get_translation_matrix,
    get_rotation_matrix_x,
    get_rotation_matrix_y,
    get_rotation_matrix_z,
)


class Node:
    def __init__(
        self,
        name: str,
        offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        parent: Optional[Node] = None,
        limits: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        """
        Args:
            name (str): Identifier for the joint
            offset: (x, y, z) distance from parent joint
            parent: The parent Node object (or None for root)
            limits: Constraints
        """
        self.name: str = name
        self.parent: Optional[Node] = parent
        self.children: List[Node] = []

        self.offset: NDArray[np.float32] = np.array(offset, dtype=np.float32)
        self.rotation: NDArray[np.float32] = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.constraints: Dict[str, Tuple[float, float]]
        if limits:
            self.constraints = limits
        else:
            self.constraints = {
                "x": (-180.0, 180.0),
                "y": (-180.0, 180.0),
                "z": (-180.0, 180.0),
            }

        self.local_matrix: Mat4x4 = np.eye(4, dtype=np.float32)
        self.world_matrix: Mat4x4 = np.eye(4, dtype=np.float32)
        self.world_position: NDArray[np.float32] = np.zeros(3, dtype=np.float32)

        if parent:
            parent.children.append(self)

    def update_matrices(self) -> None:
        """
        Forward Kinematics Pose.
        Recursively calculates local and world matrices.
        """
        Rx = get_rotation_matrix_x(self.rotation[0])
        Ry = get_rotation_matrix_y(self.rotation[1])
        Rz = get_rotation_matrix_z(self.rotation[2])

        R_local: Mat4x4 = Rz @ Ry @ Rx
        T_local = get_translation_matrix(
            float(self.offset[0]), float(self.offset[1]), float(self.offset[2])
        )
        self.local_matrix = T_local @ R_local

        if self.parent:
            self.world_matrix = self.parent.world_matrix @ self.local_matrix
        else:
            self.world_matrix = self.local_matrix

        self.world_position = self.world_matrix[:3, 3]

        for child in self.children:
            child.update_matrices()

    def get_global_position(self) -> NDArray[np.float32]:
        return self.world_position

    def apply_rotation(self, delta_angles: NDArray[np.float32]) -> None:
        """
        Applies a delta rotation and clamps the limits.
        delta_angles: [dx, dy, dz] in degrees.
        """
        self.rotation += delta_angles
        self.clamp_rotation()

    def clamp_rotation(self) -> None:
        """
        Enforces the joint limits.
        """
        # self.rotation[0] = np.clip(
        #     self.rotation[0], self.constraints["x"][0], self.constraints["x"][1]
        # )
        self.rotation[1] = np.clip(
            self.rotation[1], self.constraints["y"][0], self.constraints["y"][1]
        )
        # self.rotation[2] = np.clip(
        #     self.rotation[2], self.constraints["z"][0], self.constraints["z"][1]
        # )

    def reset(self) -> None:
        """
        Resets rotation to 0.
        """
        self.rotation = np.zeros(3, dtype=np.float32)
        for child in self.children:
            child.reset()
