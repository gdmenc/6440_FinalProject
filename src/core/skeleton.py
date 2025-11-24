from typing import Dict, List, Optional
from src.core.node import Node

class Skeleton:
    """
    Manages the hierarchy of Nodes.
    Acts as a wrapper to provide easy access to specific joints.
    Handles global updates.
    """
    def __init__(self, root: Node) -> None:
        """
        Args:
            root: The base Node of the character (usually waist or pelvis)
        """
        self.root: Node = root
        self.joints_map: Dict[str, Node] = {}
        self.joints_list: List[Node] = []
        self._index_joints(self.root)

    def _index_joints(self, node: Node) -> None:
        """
        Recursive helper to populate joints_map and joints_list.
        """
        self.joints_map[node.name] = node
        self.joints_list.append(node)

        for child in node.children:
            self._index_joints(child)
    
    def update(self) -> None:
        """
        Triggers the forward kinematics pass for the entire body.
        Should be called once per frame, or after the IK solver runs.
        """
        self.root.update_matrices()

    def get_joint(self, name: str) -> Optional[Node]:
        """
        Retrieves a joint by name.
        Returns None if the joint does not exist.
        """
        return self.joints_map.get(name)

    def reset_pose(self) -> None:
        """
        Resets all joints in the skeleton to their default state.
        """
        self.root.reset()
        self.update()

    def get_chain(self, end_effector_name: str) -> List[Node]:
        """
        Returns the kinematic chain (list of nodes) from the root up
            to the specific end-effector.
        """
        effector = self.get_joint(end_effector_name)
        if not effector:
            return []

        chain: List[Node] = []
        current: Optional[Node] = effector

        while current is not None:
            chain.append(current)
            current = current.parent
            if current == self.root:
                chain.append(current)
                break
            
        return chain[::-1]