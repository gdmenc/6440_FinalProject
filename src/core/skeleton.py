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
        Sets arms to hang down for a natural standing pose.
        """
        self.root.reset()

        # Set arms to hang down naturally
        # Note: Geometry is mirrored, so rotation signs are swapped
        r_shoulder = self.get_joint("R_Shoulder")
        if r_shoulder:
            # Rotate down (Z-axis roll) - Right is now on screen left
            r_shoulder.rotation[2] = 80.0
            r_shoulder.clamp_rotation()

        l_shoulder = self.get_joint("L_Shoulder")
        if l_shoulder:
            # Rotate down (Z-axis roll) - Left is now on screen right
            l_shoulder.rotation[2] = -80.0
            l_shoulder.clamp_rotation()

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

    def get_chain_anchored(self, end_effector_name: str) -> List[Node]:
        """
        Returns a localized kinematic chain that starts from a limb-specific anchor
        (shoulder/hip/neck) up to the given end-effector. This improves locality of IK.
        """
        full_chain = self.get_chain(end_effector_name)
        if not full_chain:
            return []

        # Map effectors to their local anchors
        anchor_by_prefix = {
            "R_Hand": "R_Shoulder",
            "R_Wrist": "R_Shoulder",
            "R_Elbow": "R_Shoulder",
            "L_Hand": "L_Shoulder",
            "L_Wrist": "L_Shoulder",
            "L_Elbow": "L_Shoulder",
            "R_Foot": "R_Hip",
            "R_Ankle": "R_Hip",
            "R_Knee": "R_Hip",
            "L_Foot": "L_Hip",
            "L_Ankle": "L_Hip",
            "L_Knee": "L_Hip",
            "Head": "Neck",
            "Neck": "Chest",
        }

        # Determine anchor name for this effector
        anchor_name = None
        for key, value in anchor_by_prefix.items():
            if end_effector_name.startswith(key):
                anchor_name = value
                break

        if not anchor_name:
            return full_chain

        anchor_node = self.get_joint(anchor_name)
        if not anchor_node:
            return full_chain

        # Slice chain to start from the anchor
        try:
            start_idx = full_chain.index(anchor_node)
            return full_chain[start_idx:]
        except ValueError:
            return full_chain
