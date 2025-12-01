"""
Rendering module for skeleton visualization.
Supports multiple rendering modes: wireframe and capsule-based.
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from src.core.node import Node


class SkeletonRenderer:
    """
    Handles visualization of the skeleton with multiple rendering modes.
    """
    
    def __init__(self):
        self.mode = "capsule"  # "wireframe" or "capsule"
    
    def set_mode(self, mode: str):
        """
        Set the rendering mode.
        
        Args:
            mode: Either "wireframe" or "capsule"
        """
        if mode in ("wireframe", "capsule"):
            self.mode = mode
        else:
            raise ValueError(f"Unknown rendering mode: {mode}")
    
    def toggle_mode(self):
        """Toggle between wireframe and capsule modes."""
        self.mode = "capsule" if self.mode == "wireframe" else "wireframe"
        return self.mode
    
    def render(self, root_node: Node):
        """
        Render the skeleton starting from the root node.
        
        Args:
            root_node: The root node of the skeleton hierarchy
        """
        if self.mode == "capsule":
            glEnable(GL_LIGHTING)
            self._render_capsule(root_node)
            glDisable(GL_LIGHTING)
        else:
            self._render_wireframe(root_node)
    
    def _render_wireframe(self, node: Node):
        """
        Recursively draws bones and joints using wireframe visualization.
        """
        # Draw joint as a point
        glPushMatrix()
        glTranslatef(node.world_position[0], node.world_position[1], node.world_position[2])
        glColor3f(1.0, 0.2, 0.2)
        
        glPointSize(8)
        glBegin(GL_POINTS)
        glVertex3f(0, 0, 0)
        glEnd()
        glPopMatrix()
        
        # Draw bone as a line to parent
        if node.parent:
            glBegin(GL_LINES)
            glColor3f(0.9, 0.9, 0.9)
            glVertex3fv(node.parent.world_position)
            glVertex3fv(node.world_position)
            glEnd()
        
        # Recurse to children
        for child in node.children:
            self._render_wireframe(child)
    
    def _render_capsule(self, node: Node):
        """
        Recursively draws bones as capsules (spheres + cylinders).
        """
        color = self._get_body_part_color(node.name)
        joint_radius = self._get_joint_radius(node.name)
        
        # Draw sphere at joint
        self._draw_sphere(node.world_position, joint_radius, color)
        
        # Draw cylinder to parent
        if node.parent:
            parent_radius = self._get_joint_radius(node.parent.name)
            bone_radius = (joint_radius + parent_radius) / 3.0
            self._draw_cylinder_between(
                node.parent.world_position,
                node.world_position,
                bone_radius,
                color
            )
        
        # Recurse to children
        for child in node.children:
            self._render_capsule(child)
    
    @staticmethod
    def _draw_sphere(position, radius, color=(0.8, 0.6, 0.4)):
        """Draws a sphere at the given position."""
        glPushMatrix()
        glTranslatef(position[0], position[1], position[2])
        glColor3f(*color)
        
        quadric = gluNewQuadric()
        gluQuadricNormals(quadric, GLU_SMOOTH)
        gluSphere(quadric, radius, 16, 16)
        gluDeleteQuadric(quadric)
        
        glPopMatrix()
    
    @staticmethod
    def _draw_cylinder_between(start_pos, end_pos, radius, color=(0.8, 0.6, 0.4)):
        """Draws a cylinder between two points."""
        glPushMatrix()
        
        glTranslatef(start_pos[0], start_pos[1], start_pos[2])
        direction = end_pos - start_pos
        length = np.linalg.norm(direction)
        
        if length > 1e-3:
            direction = direction / length
            z_axis = np.array([0, 0, 1], dtype=np.float32)
            axis = np.cross(z_axis, direction)
            axis_len = np.linalg.norm(axis)
            
            if axis_len > 1e-3:
                axis /= axis_len
                angle = np.degrees(np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0)))
                glRotatef(angle, axis[0], axis[1], axis[2])
            elif np.dot(z_axis, direction) < 0:
                glRotatef(180, 1, 0, 0)
        
        glColor3f(*color)
        quadric = gluNewQuadric()
        gluQuadricNormals(quadric, GLU_SMOOTH)
        gluCylinder(quadric, radius, radius, length, 16, 1)
        gluDeleteQuadric(quadric)
        
        glPopMatrix()
    
    @staticmethod
    def _get_body_part_color(node_name: str) -> tuple:
        """Returns color based on body part for visual distinction."""
        name = node_name.lower()
        
        # Head - light skin tone
        if "head" in name or "neck" in name:
            return (0.95, 0.8, 0.7)
        # Torso - medium skin tone
        if "spine" in name or "chest" in name or "root" in name:
            return (0.85, 0.65, 0.5)
        # Right arm - slightly blue tint
        if "r_" in name and any(part in name for part in ("shoulder", "elbow", "wrist", "hand")):
            return (0.7, 0.75, 0.9)
        # Left arm - slightly red tint
        if "l_" in name and any(part in name for part in ("shoulder", "elbow", "wrist", "hand")):
            return (0.9, 0.7, 0.75)
        # Right leg - darker blue
        if "r_" in name and any(part in name for part in ("hip", "knee", "ankle", "foot")):
            return (0.6, 0.65, 0.85)
        # Left leg - darker red
        if "l_" in name and any(part in name for part in ("hip", "knee", "ankle", "foot")):
            return (0.85, 0.6, 0.65)
        # Default
        return (0.8, 0.6, 0.4)
    
    @staticmethod
    def _get_joint_radius(node_name: str) -> float:
        """Returns appropriate radius for different body parts."""
        name = node_name.lower()
        
        if "head" in name:
            return 0.7
        if "neck" in name:
            return 0.4
        if "chest" in name or "spine2" in name:
            return 0.6
        if "spine" in name or "root" in name:
            return 0.5
        if "shoulder" in name or "hip" in name:
            return 0.4
        if "elbow" in name or "knee" in name:
            return 0.3
        if "wrist" in name or "ankle" in name:
            return 0.25
        if "hand" in name or "foot" in name:
            return 0.2
        return 0.3


