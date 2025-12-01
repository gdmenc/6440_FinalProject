import re
import numpy as np
from typing import Optional
from src.core.skeleton import Skeleton


class CommandResult:
    """
    Data Transfer Object to return parsing results cleanly.
    """

    def __init__(
        self,
        success: bool,
        message: str,
        effector_name: Optional[str] = None,
        target_pos: Optional[np.ndarray] = None,
    ):
        self.success = success
        self.message = message
        self.effector_name = effector_name
        self.target_pos = target_pos


class Commander:
    def __init__(self, skeleton: Skeleton):
        self.skeleton = skeleton

        self.part_aliases = {
            # Head and neck
            "head": "Head",
            "neck": "Neck",
            # Torso
            "chest": "Chest",
            "torso": "Chest",
            "spine": "Spine2",
            # Right arm
            "right hand": "R_Hand",
            "right wrist": "R_Wrist",
            "right elbow": "R_Elbow",
            "right shoulder": "R_Shoulder",
            # Left arm
            "left hand": "L_Hand",
            "left wrist": "L_Wrist",
            "left elbow": "L_Elbow",
            "left shoulder": "L_Shoulder",
            # Right leg
            "right foot": "R_Foot",
            "right ankle": "R_Ankle",
            "right knee": "R_Knee",
            "right hip": "R_Hip",
            # Left leg
            "left foot": "L_Foot",
            "left ankle": "L_Ankle",
            "left knee": "L_Knee",
            "left hip": "L_Hip",
            # Generic (defaults to right)
            "hand": "R_Hand",
            "wrist": "R_Wrist",
            "elbow": "R_Elbow",
            "shoulder": "R_Shoulder",
            "foot": "R_Foot",
            "ankle": "R_Ankle",
            "knee": "R_Knee",
            "hip": "R_Hip",
        }

        self.direction_map = {
            "up": np.array([0, 1, 0], dtype=np.float32),
            "raise": np.array([0, 1, 0], dtype=np.float32),
            "lift": np.array([0, 1, 0], dtype=np.float32),
            "down": np.array([0, -1, 0], dtype=np.float32),
            "lower": np.array([0, -1, 0], dtype=np.float32),
            "drop": np.array([0, -1, 0], dtype=np.float32),
            "left": np.array([1, 0, 0], dtype=np.float32),
            "right": np.array([-1, 0, 0], dtype=np.float32),
            "forward": np.array([0, 0, 1], dtype=np.float32),
            "back": np.array([0, 0, -1], dtype=np.float32),
            "backward": np.array([0, 0, -1], dtype=np.float32),
        }

    def parse(self, command_text: str):
        text = command_text.lower().strip()
        if text == "reset":
            self.skeleton.reset_pose()
            return CommandResult(True, "Skeleton reset to default pose.")

        target_joint_name = None

        sorted_aliases = sorted(self.part_aliases.keys(), key=len, reverse=True)
        for alias in sorted_aliases:
            if alias in text:
                target_joint_name = self.part_aliases[alias]
                text = text.replace(alias, "")
                break

        if not target_joint_name:
            return CommandResult(False, "Could not identify a valid body part!")

        node = self.skeleton.get_joint(target_joint_name)
        if not node:
            return CommandResult(
                False, f"Joint '{target_joint_name}' not found in skeleton!"
            )

        direction_vector = np.array([0, 0, 0], dtype=np.float32)

        found_action = False
        for keyword, vector in self.direction_map.items():
            if keyword in text:
                direction_vector = vector
                found_action = True
                break

        if not found_action:
            return CommandResult(False, "Could not understand action!")

        amount = 2.0
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if numbers:
            try:
                amount = float(numbers[-1])
            except ValueError:
                pass

        current_pos = node.get_global_position()
        displacement = direction_vector * amount
        target_pos = current_pos + displacement

        return CommandResult(
            success=True,
            message=f"Moving {target_joint_name}...",
            effector_name=target_joint_name,
            target_pos=target_pos,
        )
