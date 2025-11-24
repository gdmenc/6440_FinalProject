import numpy as np
from numpy.typing import NDArray
from typing import List

from src.core.skeleton import Skeleton
from src.core.node import Node

class JacobianIK:
    def __init__(self, 
                damping: float = 0.1, 
                step_size: float = 0.5, 
                max_iterations: int = 10,
                threshold: float = 0.1
    ) -> None:
        self.damping = damping
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.threshold = threshold

    def solve(self, skeleton: Skeleton, effector_name: str, target_pos: NDArray[np.float32]) -> bool:
        """
        Main IK Loop. Modifies the skeleton's joints in-place.
        Returns: True if target reached, False otherwise
        """
        chain = skeleton.get_chain(effector_name)
        if not chain:
            print(f"ERROR: Effector name {effector_name} not found!")
            return False

        end_effector = chain[-1]

        for _ in range(self.max_iterations):
            current_pos = end_effector.get_global_position()
            error_vector = target_pos - current_pos
            
            if np.sum(error_vector**2) < (self.threshold**2):
                return True

            J = self._compute_jacobian(chain, current_pos)
            J_J_T = J @ J.T
            damping_matrix = (self.damping ** 2) * np.eye(3)

            try:
                inverse_term = np.linalg.inv(J_J_T + damping_matrix)
            except np.linalg.LinAlgError:
                return False

            delta_theta = J.T @ inverse_term @ error_vector
            self._apply_deltas(chain, delta_theta)
            skeleton.update()

        return False

    def _compute_jacobian(self, chain: List[Node], effector_pos: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Constructs the Jacobian matrix.
        """
        degrees_of_freedom = len(chain) * 3
        J = np.zeros((3, degrees_of_freedom), dtype=np.float32)
        col_idx = 0

        for node in chain:
            node_pos = node.get_global_position()
            r = effector_pos - node_pos

            if node.parent:
                parent_rot = node.parent.world_matrix[:3, :3]
            else:
                parent_rot = np.eye(3, dtype=np.float32)

            axis_x = parent_rot[:, 0]
            J[:, col_idx] = np.cross(axis_x, r)
            col_idx += 1

            axis_y = parent_rot[:, 1]
            J[:, col_idx] = np.cross(axis_y, r)
            col_idx += 1

            axis_z = parent_rot[:, 2]
            J[:, col_idx] = np.cross(axis_z, r)
            col_idx += 1

        return J

    def _apply_deltas(self, chain: List[Node], delta_theta: NDArray[np.float32]) -> None:
        """
        Maps the flat delta_theta vector back to specific joints.
        """
        idx = 0
        for node in chain:
            dx = delta_theta[idx] * self.step_size
            dy = delta_theta[idx + 1] * self.step_size
            dz = delta_theta[idx + 2] * self.step_size
            angle_change = np.array([np.degrees(dx), np.degrees(dy), np.degrees(dz)])
            node.apply_rotation(angle_change)
            idx += 3