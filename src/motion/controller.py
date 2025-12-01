"""
Motion controller system for procedural animations.
Provides base class and implementations for cyclic motions like walking.
"""
import numpy as np
from typing import Optional, Dict
from abc import ABC, abstractmethod
from src.core.skeleton import Skeleton


class MotionController(ABC):
    """
    Abstract base class for motion controllers.
    Subclasses implement specific motions (walk, wave, etc.)
    """

    def __init__(self, skeleton: Skeleton):
        self.skeleton = skeleton
        self.active = False
        self.time = 0.0

    def start(self):
        """Start the motion."""
        self.active = True
        self.time = 0.0

    def pause(self):
        """Pause the motion without resetting pose."""
        self.active = False

    def stop(self):
        """Stop the motion and reset to neutral pose."""
        self.active = False
        self.skeleton.reset_pose()
        self.skeleton.update()

    def is_active(self) -> bool:
        return self.active

    @abstractmethod
    def update(self, dt: float) -> None:
        """
        Update the motion by dt seconds.
        Should modify skeleton joint rotations directly.
        """
        pass


class WalkMotion(MotionController):
    """
    Procedural walking animation using sinusoidal joint rotations.
    
    Animates:
    - Hips: alternating flexion/extension
    - Knees: bend during swing phase
    - Ankles: plantarflexion/dorsiflexion
    - Shoulders: arm swing opposite to legs
    - Elbows: slight bend during forward swing
    - Spine: subtle counter-rotation
    """

    def __init__(self, skeleton: Skeleton, speed: float = 1.0):
        super().__init__(skeleton)
        self.speed = speed
        self.cycle_duration = 1.2
        
        # Motion amplitudes (in degrees)
        self.params = {
            # Legs
            "hip_flexion": 35.0,      # Forward/back swing of thigh
            "knee_flexion": 45.0,     # Knee bend during swing
            "ankle_flexion": 15.0,    # Foot tilt
            
            # Arms (swing opposite to legs)
            "shoulder_swing": 25.0,   # Forward/back arm swing
            "elbow_bend": 20.0,       # Elbow flexion during swing
            
            # Spine
            "spine_rotation": 8.0,    # Counter-rotation with legs
            "spine_side_bend": 3.0,   # Subtle side-to-side
        }

    def update(self, dt: float) -> None:
        if not self.active:
            return

        self.time += dt * self.speed        
        phase = (self.time / self.cycle_duration) * 2 * np.pi
        
        joints = self._get_joints()
        if not joints:
            return

        r_hip_angle = self.params["hip_flexion"] * np.sin(phase)
        l_hip_angle = self.params["hip_flexion"] * np.sin(phase + np.pi)
        
        if joints["r_hip"]:
            joints["r_hip"].rotation[0] = r_hip_angle
            joints["r_hip"].clamp_rotation()
        if joints["l_hip"]:
            joints["l_hip"].rotation[0] = l_hip_angle
            joints["l_hip"].clamp_rotation()

        r_knee_phase = phase - np.pi / 4  # Offset for timing
        l_knee_phase = phase + np.pi - np.pi / 4
        
        r_knee_angle = -self.params["knee_flexion"] * max(0, np.sin(r_knee_phase))
        l_knee_angle = -self.params["knee_flexion"] * max(0, np.sin(l_knee_phase))
        
        if joints["r_knee"]:
            joints["r_knee"].rotation[0] = r_knee_angle
            joints["r_knee"].clamp_rotation()
        if joints["l_knee"]:
            joints["l_knee"].rotation[0] = l_knee_angle
            joints["l_knee"].clamp_rotation()

        r_ankle_angle = self.params["ankle_flexion"] * np.sin(phase + np.pi / 2)
        l_ankle_angle = self.params["ankle_flexion"] * np.sin(phase + np.pi + np.pi / 2)
        
        if joints["r_ankle"]:
            joints["r_ankle"].rotation[0] = r_ankle_angle
            joints["r_ankle"].clamp_rotation()
        if joints["l_ankle"]:
            joints["l_ankle"].rotation[0] = l_ankle_angle
            joints["l_ankle"].clamp_rotation()

        r_shoulder_swing = -self.params["shoulder_swing"] * np.sin(phase + np.pi)
        l_shoulder_swing = -self.params["shoulder_swing"] * np.sin(phase)
        
        if joints["r_shoulder"]:
            joints["r_shoulder"].rotation[0] = r_shoulder_swing
            joints["r_shoulder"].clamp_rotation()
        if joints["l_shoulder"]:
            joints["l_shoulder"].rotation[0] = l_shoulder_swing
            joints["l_shoulder"].clamp_rotation()

        r_elbow_bend = -self.params["elbow_bend"] * max(0, -np.sin(phase + np.pi))
        l_elbow_bend = self.params["elbow_bend"] * max(0, -np.sin(phase))
        
        if joints["r_elbow"]:
            joints["r_elbow"].rotation[1] = r_elbow_bend
            joints["r_elbow"].clamp_rotation()
        if joints["l_elbow"]:
            joints["l_elbow"].rotation[1] = l_elbow_bend
            joints["l_elbow"].clamp_rotation()

        spine_rot_y = self.params["spine_rotation"] * np.sin(phase)
        spine_rot_z = self.params["spine_side_bend"] * np.sin(phase * 2)
        
        if joints["spine1"]:
            joints["spine1"].rotation[1] = spine_rot_y * 0.3
            joints["spine1"].rotation[2] = spine_rot_z * 0.3
            joints["spine1"].clamp_rotation()
        if joints["spine2"]:
            joints["spine2"].rotation[1] = spine_rot_y * 0.4
            joints["spine2"].rotation[2] = spine_rot_z * 0.4
            joints["spine2"].clamp_rotation()
        if joints["chest"]:
            joints["chest"].rotation[1] = spine_rot_y * 0.3
            joints["chest"].rotation[2] = spine_rot_z * 0.3
            joints["chest"].clamp_rotation()

        if joints["neck"]:
            joints["neck"].rotation[1] = -spine_rot_y * 0.5
            joints["neck"].clamp_rotation()

        self.skeleton.update()

    def _get_joints(self) -> Dict[str, Optional[any]]:
        """Retrieve all joints needed for walking animation."""
        return {
            # Legs
            "r_hip": self.skeleton.get_joint("R_Hip"),
            "l_hip": self.skeleton.get_joint("L_Hip"),
            "r_knee": self.skeleton.get_joint("R_Knee"),
            "l_knee": self.skeleton.get_joint("L_Knee"),
            "r_ankle": self.skeleton.get_joint("R_Ankle"),
            "l_ankle": self.skeleton.get_joint("L_Ankle"),
            # Arms
            "r_shoulder": self.skeleton.get_joint("R_Shoulder"),
            "l_shoulder": self.skeleton.get_joint("L_Shoulder"),
            "r_elbow": self.skeleton.get_joint("R_Elbow"),
            "l_elbow": self.skeleton.get_joint("L_Elbow"),
            # Spine
            "spine1": self.skeleton.get_joint("Spine1"),
            "spine2": self.skeleton.get_joint("Spine2"),
            "chest": self.skeleton.get_joint("Chest"),
            "neck": self.skeleton.get_joint("Neck"),
        }

    def set_speed(self, speed: float) -> None:
        """Adjust animation speed (1.0 = normal, 2.0 = double speed)."""
        self.speed = max(0.1, min(3.0, speed))

